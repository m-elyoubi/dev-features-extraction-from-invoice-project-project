import json
import boto3
import io
import cv2 as cv
from skimage.metrics import structural_similarity as ssim
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from fpdf import FPDF
import time
from datetime import datetime
import re
import os
import csv
import concurrent.futures
import fitz
from io import BytesIO
from PIL import Image
import locale


# intialization 
s3 = boto3.client('s3')
textract_client = boto3.client('textract')
# s3_bucket = os.environ.get('s3_bucket')
# images_folder = os.environ.get('images_folder')
# split_prefix=os.environ.get('split_prefix')
# SaveInvoicesprefix="SaveInvoices"
# output_bucket = os.environ.get('output_bucket')


def save_images_to_pdf(image_paths, output_pdf_path):
    try:
        pdf = FPDF()
        
        for image_path in image_paths:
            pdf.add_page()
            pdf.image(image_path, 0, 0, 210, 297)  # Adjust the dimensions as needed

        pdf.output(output_pdf_path)

    except Exception as e:
        print(f"Error saving images to PDF: {str(e)}")

def save_images_to_pdf_and_upload(np_array, s3_bucket, destination_prefix, pdf_name, temp_pdf_name):
    try:
        image_paths = []
        for i, np_img in enumerate(np_array):
            cv_img_temp = cv.cvtColor(np_img, cv.COLOR_RGB2BGR)
            _, img_bytes = cv.imencode('.jpg', cv_img_temp)
            temp_img_path = f"/tmp/image_{i}.jpg"
            with open(temp_img_path, 'wb') as f:
                f.write(img_bytes)
            image_paths.append(temp_img_path)
        
    
        save_images_to_pdf(image_paths, temp_pdf_name)

        # Upload the PDF to S3
        pdf_output_key = f"{destination_prefix}{pdf_name}"
        with open(temp_pdf_name, 'rb') as pdf_file:
            s3.upload_fileobj(pdf_file, s3_bucket, pdf_output_key)

        print(f"Uploaded PDF to S3: s3://{s3_bucket}/{pdf_output_key}")

    except Exception as e:
        print(f"Error saving images to PDF and uploading to S3: {str(e)}")

def extract_text_from_image(s3_bucket,image_key):

    response = textract_client.start_document_text_detection(
        DocumentLocation={'S3Object': {'Bucket': s3_bucket, 'Name': image_key}}
    )

    # Get the JobId from the response
    job_id = response['JobId']

    # Wait for the analysis job to complete
    while True:
        response = textract_client.get_document_text_detection(JobId=job_id)
        status = response['JobStatus']

        if status in ['SUCCEEDED', 'FAILED']:
            break

    # Retrieve the extracted text
    if status == 'SUCCEEDED':
        blocks = response['Blocks']
        extracted_text = ''
        for block in blocks:
            if block['BlockType'] == 'LINE':
                extracted_text += block['Text'] + '\n'

        return extracted_text
    else:
        print("Textract job failed.")

def extract_account_number(s3_bucket,image_key):
    text=extract_text_from_image(s3_bucket,image_key)
    due_date_pattern = re.compile(r'(Cigna.|\n|Bill Date:\s*|TOTAL AMOUNT DUE|Payment Due Date|Due Date|DUE DATE)\s+((\d{1,2}/\d{1,2}/\d{2,4})|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{2,4})')
    due_date_match = due_date_pattern.search(text)
    formatted_date = due_date_match.group(2) if due_date_match else None
    return formatted_date


def process_pages(s3_bucket,file_content_array,images_keys_np,split_prefix):
    try:
        reference_roi_gray = None
        reference_account_number = None
        similar_pages = []
        pdf_index = 1
        temp_pdf_name=None 
        for i, file_content in enumerate(file_content_array, start=1):
            img_np= cv.imdecode(np.fromstring(file_content, np.uint8), cv.IMREAD_COLOR)
            max_display_height = 800
            max_display_width = 1200
            height, width = img_np.shape[:2]
            if height > max_display_height or width > max_display_width:
                scale_factor = min(max_display_height / height, max_display_width / width)
                img = cv.resize(img_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA)

            start_x, end_x, start_y, end_y = 0,250,0,70   #0, 180, 0, 80

            roi = img_np[start_y:end_y, start_x:end_x]
            img = cv.fastNlMeansDenoisingColored(roi, None, 10, 10, 7, 21)

            roi_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            current_account_number = extract_account_number(s3_bucket,images_keys_np[i])
            pdf_name = f"pdf_{pdf_index}.pdf"
            temp_pdf_name = f"/tmp/pdf_{pdf_index}.pdf"
            # _, result_threshold = cv.threshold(roi_gray, 130, 255, cv.THRESH_BINARY)  # the first way
            if reference_roi_gray is not None or current_account_number is not None:
                # Compare the current ROI with the reference ROI using SSIM
                similarity_index = ssim(reference_roi_gray, roi_gray)
                print(f"Similarity Index between Page {i} and Page {i-1}: {similarity_index}")

                # similarity_threshold =0.58  if i == 7 else 0.90
                similarity_threshold = 0.90

                if similarity_index >= similarity_threshold or  current_account_number == reference_account_number:
                    print(f"Page {i} is similar to the previous page. Adding to the current sequence.")
                    similar_pages.append(img_np)   
                else:
                    # Pages are not similar, save the current sequence to a new PDF
                    if similar_pages:
                        print(f"Pages are not similar similar_pages")
                        save_images_to_pdf_and_upload(similar_pages,s3_bucket,split_prefix,pdf_name,temp_pdf_name)
                        print(f"Saved similar pages to {temp_pdf_name}")
                        pdf_index += 1
                    
                    # Start a new sequence with the current page
                    similar_pages = [img_np]
            else:
                save_images_to_pdf_and_upload([img_np],s3_bucket,split_prefix,pdf_name,temp_pdf_name)
                print(f"Saved the first page to {pdf_name}")
                pdf_index += 1

            reference_roi_gray = roi_gray
            reference_account_number = current_account_number

        # Save the last sequence to a new PDF if not empty
        if similar_pages:
            print(f"Save the last sequence to a new PDF if not emptyL")
            save_images_to_pdf_and_upload(similar_pages,s3_bucket,split_prefix,pdf_name,temp_pdf_name)
            print(f"Saved the last  pages to {temp_pdf_name}")

    except Exception as e:
        print(f"Error processing pages: {str(e)}")

def start_textract_job(object_key, s3_bucket):
    response = textract_client.start_document_text_detection(
        DocumentLocation={'S3Object': {'Bucket': s3_bucket, 'Name': object_key}}
    )
    return response['JobId']

def wait_for_textract_completion(job_id):
    try:
        while True:
            response = textract_client.get_document_text_detection(JobId=job_id)

            status = response['JobStatus']
            # print(f"Textract Job Status: {status}")

            if status in ['SUCCEEDED', 'FAILED', 'PARTIAL_SUCCESS']:
                return response
            elif status == 'IN_PROGRESS':
                time.sleep(5)  # Adjust the sleep duration as needed
            else:
                raise ValueError(f"Unexpected job status: {status}")

    except Exception as e:
        print(f"Error in wait_for_textract_completion: {str(e)}")

def pdf_to_text(pdf_key, s3_bucket):
    job_id = start_textract_job(pdf_key, s3_bucket)

    try:
        wait_for_textract_completion(job_id)
        results = textract_client.get_document_text_detection(JobId=job_id)
        # print(f"Textract Results: {results}")

        text = "\n".join(result['Text'] for result in results['Blocks'] if result['BlockType'] == 'LINE')
        return text
    except Exception as e:
        print(f"Error extracting content from PDF {pdf_key}: {str(e)}")

    return None

def rename_s3_file(bucket_name, old_key, new_key):
    s3.copy_object(Bucket=bucket_name, CopySource={'Bucket': bucket_name, 'Key': old_key}, Key=new_key)
    s3.delete_object(Bucket=bucket_name, Key=old_key)

def process_single_image(image_key, s3_bucket, conditions):
    job_id = start_textract_job(image_key, s3_bucket)

    try:
        wait_for_textract_completion(job_id)
        results = textract_client.get_document_text_detection(JobId=job_id)
        text = "\n".join(result['Text'] for result in results['Blocks'] if result['BlockType'] == 'LINE')

        if any(condition in text for condition in conditions) and 'Dear' not in text:
            return image_key

    except Exception as e:
        print(f"Error processing single image {image_key}: {str(e)}")

    return None

def process_images(images, s3_bucket):
    try:   
        extracted_images = []
        conditions = ['Account Number','Customer Number', 'Loan Number','Customer ID', 'LOAN #']

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_single_image, image_key, s3_bucket, conditions) for image_key in images]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    extracted_images.append(result)

        return extracted_images

    except Exception as e:
        print(f"Error processing images: {str(e)}")


def convert_date(input_date):
    try:
        parsed_date = datetime.strptime(input_date, "%B %d, %Y")
    except ValueError:
        try:
            parsed_date = datetime.strptime(input_date, "%B %d,%Y")
        except ValueError:
            parsed_date = datetime.strptime(input_date, "%m/%d/%Y")
    
    output_date = parsed_date.strftime("%y%m%d")
    
    return output_date

def extract_due_date(text):
    due_date_pattern = re.compile(r'(Cigna.|\n|Bill Date:\s*|TOTAL AMOUNT DUE|Payment Due Date|Due Date|DUE DATE)\s+((\d{1,2}/\d{1,2}/\d{2,4})|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{2,4})')
    due_date_match = due_date_pattern.search(text)
    formatted_date = due_date_match.group(2) if due_date_match else None
    print(f"formatted_date before:{formatted_date}")
    date=None
    if formatted_date :
        date=convert_date(formatted_date)

    return date

def findelement(customer_name_match):
    text=customer_name_match.group(1).replace('\n',' ')
    colon_index = text.find(':')

    if colon_index != -1:
        result = text[colon_index + 1:].strip().strip("'")
    else:
        result = text
    return  result
    

# def extract_customer_name(text):
#     customer_name_pattern1 = re.compile(r'((?:.*\n){2})(PO|P.O|P.O.)\s*(BOX|Box) 853')
#     customer_name_pattern2 = re.compile(r'(.*)(PO|P.O|P.O.)\s*(BOX|Box) 853')
    

#     customer_name_match1 = customer_name_pattern1.search(text)
#     customer_name_match2 = customer_name_pattern2.search(text)

#     if customer_name_match2 is not None or customer_name_match1 is not None:
#         if customer_name_match1 and customer_name_match1.group(1).strip() != "":
#             return findelement(customer_name_match1)
#         elif customer_name_match2 and customer_name_match2.group(1).strip() != "":
#             return findelement(customer_name_match2)
#     else:
#         return None
def extract_customer_name(text):

    customer_name_match1 = re.search(r'(.+?)\n(.+?)\n(.+?)P\.O\. Box 853|((?:.*\n){3})(PO|P.O|P.O.)\s*(BOX|Box) 853',text)

    if customer_name_match1 is not None :
        x=next((x for x in customer_name_match1.groups() if x is not None), "default") 
        # if customer_name_match1 and customer_name_match1.group(1).strip()!= "":
        print(f'customer_name before:{x}')
        customer_name = ' '.join(re.findall(r'\b[A-Z][A-Z\s]+\b', x))

        if len(customer_name)>3:
            c=  customer_name
        else:
            c=  x
    
        print(f'customer_name:{c}')
        return c
            # return findelement(potential_names)
                    
    else:
        return None

# dest format: date amount payableTo payableBy ===> 231001 $9485.49 WELLS FARGO Glen Breagha LLC


def format_currency(amount):
    locale.setlocale(locale.LC_ALL, '')
    formatted_currency = locale.currency(amount, grouping=True)
    return formatted_currency

def extract_total_amount(text):

    match = re.search(r'TOTAL AMOUNT DUE\s+\d{2}/\d{2}/\d{4}\s+\$([\d.]+)|Total Payment Due\n?.*?\n?([\d,]+\.\d+)|TOTAL DUE\s*\$([\d.]+)|AMOUNT DUE\s*([\d,.]+)|Total Due\s*\$([\d,.]+)', text)
    
    if match:
        x=next((x for x in match.groups() if x is not None), "default") 

        if ',' in x:
            amount_as_float = float(x.replace(',', '')) 
        else:
            amount_as_float = float(x) 
    

    return amount_as_float

def extract_information(s3_bucket, file_obj, SaveInvoicesprefix, text):
    customer_name = extract_customer_name(text)
    date = extract_due_date(text)
    total_amount= extract_total_amount(text)
    


    dest = None

    if date is not None and customer_name is not None and total_amount:
        sanitized_customer_name = re.sub(r'[\\/:"*?<>|]', '_', customer_name)
        dest = f"{date} {format_currency(total_amount)} {sanitized_customer_name}.pdf"

    elif date and total_amount:
        sanitized_customer_name = re.sub(r'[\\/:"*?<>|]', '_', customer_name) if customer_name else "Unknown_Customer"
        dest = f"{date} {format_currency(total_amount)}.pdf"
    else:
        print(f'customer_name {customer_name}')
        print(f'total_amount {total_amount}')
        print(f"date:{date}")
 
    rename_s3_file(s3_bucket, file_obj, dest)

    return {
        'renamepdf': dest,
        'Due Date': date,
        'Customer Name': customer_name
    }

def check_s3_object_exists(bucket_name, object_key):

    try:
        s3.head_object(Bucket=bucket_name, Key=object_key)
        return True  # Object exists
    except Exception as e:
        if e.response['Error']['Code'] == '404':
            return False  # Object does not exist
        else:
            raise

def write_content_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def upload_to_s3(bucket, key, data, content_type):
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")

#DUE DATE:{DUE DATE\n(\w{3} \d{1,2}, \d{4})|Payment Due Date\n(\d{2}/\d{2}/\d{4})|DUE DATE(\d{2}/\d{1,2}/\d{4})|Invoice Date\n(\d{2}/\d{1,2}/\d{4}),Bill Date: (\w+ \d{1,2},\s*\d{4})|Cigna.\n(\w+ \d{1,2},\s*\d{4})|DUE DATE\n(\d{2}/\d{1,2}/\d{2,4})}
# Payment Due Date.*?(\d{1,2}/\d{1,2}/\d{2,4})
def extract_invoice_info(s3_bucket,pdf_file,text):
   
    patterns = {
        "Customer/client": r'((?:.*\n){2})(PO|P.O|P.O.)\s*(BOX|Box) 853|(.*)(PO|P.O|P.O.)\s*(BOX|Box) 853',
        # "Policy Number/ Account Number": r'Account Number:(?:\n\D*\d\D*\d\n\w*\n)?(\d+)|Loan Number:\s*(\d+)|Customer Number\s*\n?(\d{3,})|Account Number: (\d{3}-\d{3}-\d{3}-\d{4}-\d{2})|Customer ID:?\s*(\d*)|LOAN #\s*(\d*)|Account Number:\s*(\d*)',
        "Due Date": r'DUE DATE\n(\w{3} \d{1,2}, \d{4})|Payment Due Date\n(\d{2}/\d{2}/\d{4})|DUE DATE(\d{2}/\d{1,2}/\d{4})|Invoice Date\n(\d{2}/\d{1,2}/\d{4})|Bill Date: (\w+ \d{1,2},\s*\d{4})|Cigna.\n(\w+ \d{1,2},\s*\d{4})|DUE DATE\n(\d{2}/\d{1,2}/\d{2,4})',
        "Total Due": r'TOTAL AMOUNT DUE\s+\d{2}/\d{2}/\d{4}\s+\$([\d.]+)|Total Payment Due\n?.*?\n?([\d,]+\.\d+)|TOTAL DUE\s*\$([\d.]+)|AMOUNT DUE\s*([\d,.]+)|Total Due\s*\$([\d,.]+)',
        "company name ": r'PAYABLE TO:\s+00110?\s*(\w+\s?\w+)',
        "Property Address/Service Address": r'Property Address:\n(.+)|Service Address\s+?Page?\s+?1?\s+?of?\s+?2?\n\w*\n\w*\n(\w{1,4}\s+\w*\s+\w*)',
    }
    extracted_info = {
        "Link to PDF": f's3://{s3_bucket}/{pdf_file}'
        
    }
    for field, pattern in patterns.items():
        match = re.search(pattern, text)
        
        if match:
            x=next((x for x in match.groups() if x is not None), "default")  
            print(f"x:{x}")
            extracted_info[field] = x
        else:
            extracted_info[field] = ''  # Set a default value for missing fields

    return extracted_info	

def process_single_pdf(s3_bucket, pdf_file, output_bucket,key_name, all_csv_data, header_written):
    text_data = pdf_to_text(pdf_file, s3_bucket)
    extracted_info = extract_invoice_info(s3_bucket, pdf_file, text_data)
    csv_data = list(extracted_info.values())

    if not header_written:
        # Write header only if it hasn't been written before
        header = list(extracted_info.keys())
        all_csv_data.append(header)
        header_written = True

    all_csv_data.append(csv_data)

    # Save data to a single CSV file
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer)
    csv_writer.writerows(all_csv_data)

    s3.put_object(
        Bucket=output_bucket,
        Key=key_name,
        Body=csv_buffer.getvalue()
    )
    
def download_file(bucket, key, local_path):
    try:
        s3.download_file(bucket, key, local_path)
    except Exception as e:
        print(f"Error downloading file {key} from bucket {bucket}: {str(e)}")

def convert_page_to_image(pdf_path, page_number):
    try:
        with fitz.open(pdf_path) as pdf_document:
            pdf_page = pdf_document[page_number - 1]
            
            # Set the resolution for better quality
            zoom_factor = 2.0  
            pix = pdf_page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))
            
            # Create a PIL Image from the pixmap
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert the image to JPEG format with better quality settings
            jpeg_buffer = BytesIO()
            pil_image.save(jpeg_buffer, format='JPEG', quality=95) 
            
            return jpeg_buffer
    except Exception as e:
        print(f"Error converting page to image: {str(e)}")

def pdf_to_images(pdf_path, predix,s3_bucket):
    try:
        images = []

        for i in range(1, len(fitz.open(pdf_path)) + 1):
            image_buffer = convert_page_to_image(pdf_path, i)
            image_key = f'{predix}/page_{i}.jpg'  # Save in /tmp/ directory
            upload_to_s3(s3_bucket, image_key, image_buffer.getvalue(), 'image/jpeg')
            images.append(image_key)
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {str(e)}")

def handler(event, context):
    
    global s3_bucket
    s3_bucket = 'invoicesgranvilleinternational'
    images_folder = 'images1'
    split_prefix = 'splitpdf1/'
    SaveInvoicesprefix = "SaveInvoices1"
    output_bucket = 'invoicesoutput'

    objects = s3.list_objects(Bucket=s3_bucket)
    for obj in objects.get('Contents', []):
        pdf_file = obj['Key']
        if pdf_file.lower().endswith('.pdf'):
            local_pdf_path = f'/tmp/{pdf_file}'
            download_file(s3_bucket, pdf_file, local_pdf_path)
            print(f'Processing PDF: {local_pdf_path}')

            sorted_images = sorted(pdf_to_images(local_pdf_path,images_folder,s3_bucket), key=lambda x: int(x.split('_')[-1].split('.')[0]))            
                
            extracted_images = sorted(process_images(sorted_images, s3_bucket), key=lambda x: int(x.split('_')[-1].split('.')[0]))
            print(f"extracted_images:{extracted_images}")
            if extracted_images:
                local_image_paths = []
                images_keys_np=[]
                for image_key in extracted_images:
                    local_image_path = f"/tmp/{os.path.basename(image_key)}"
                    download_file(s3_bucket, image_key, local_image_path)
                    local_image_paths.append(local_image_path)
                    images_keys_np.append(image_key)
                
                image_contents = []
                print(f"local_image_path:{local_image_paths}")
                for local_image_path in local_image_paths:
                    with open(local_image_path, 'rb') as image_file:
                        image_content = image_file.read()
                        image_contents.append(image_content)
                
                # Save images to PDF
                pdf_output_path_temp = '/tmp/output.pdf'
                save_images_to_pdf(local_image_paths, pdf_output_path_temp)
                pdf_output_key = 'pdfbasedoncondition/newresult.pdf'  # Adjust the S3 key as needed
                # Upload the PDF to S3
                upload_to_s3(s3_bucket, pdf_output_key, open(pdf_output_path_temp, 'rb'), 'application/pdf')
                print(f'PDF created and uploaded successfully: s3://{s3_bucket}/{pdf_output_key}')
                print(f"image_contents:{image_contents}")
                if image_contents:
                    process_pages(s3_bucket,image_contents,images_keys_np,split_prefix)
                
    return {
        'statusCode': 200,
        'body': 'Textract jobs completed for all PDFs in the input bucket.'
    }
