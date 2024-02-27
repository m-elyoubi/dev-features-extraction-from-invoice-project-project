# Import libraries 
import json
import boto3
import io
import cv2 as cv
from skimage.metrics import structural_similarity as ssim
import numpy as np
from fpdf import FPDF
import time
import re
import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz
from io import BytesIO
from PIL import Image
from datetime import datetime
import locale
from dateutil import parser
import pandas as pd

# Initialize AWS clients
s3 = boto3.client('s3')  # S3 client
textract_client = boto3.client('textract')  # Textract client
# Define constants and environment variables (commented out)
images_folder = os.environ.get('images_folder')
split_prefix = os.environ.get('split_prefix')
SaveInvoices_prefix = os.environ.get('SAVEINVOICESPREFIX')
output_bucket = os.environ.get('output_bucket')
input_s3_bucket = os.environ.get('input_s3_bucket')
invoices_s3_bucket = os.environ.get('invoices_s3_bucket')
file_key = os.environ.get('FILE_COMPANYNAME')
images_folder = os.environ.get('IMAGES_FOLDER')


def save_images_to_pdf(image_paths, output_pdf_path):
    """
    Save a list of images into a single PDF file.

    Parameters:
    - image_paths: List of file paths to the images.
    - output_pdf_path: File path for the output PDF.

    Returns:
    None
    """
    try:
        pdf = FPDF()

        for image_path in image_paths:
            pdf.add_page()
            pdf.image(image_path, 0, 0, 210, 297)  # Adjust the dimensions as needed

        pdf.output(output_pdf_path)

    except Exception as e:
        print(f"Error saving images to PDF: {str(e)}")

def save_images_to_pdf_and_upload(np_array, s3_bucket, destination_prefix, pdf_name, temp_pdf_name):
    """
    Save images as PDF and upload to S3.

    Parameters:
    - np_array: NumPy array of images.
    - s3_bucket: S3 bucket to upload the PDF to.
    - destination_prefix: Prefix for the destination key in S3.
    - pdf_name: Name of the PDF file.
    - temp_pdf_name: Temporary file path for the PDF.

    Returns:
    None
    """
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
        pdf_output_key = f"{destination_prefix}/{pdf_name}"
        with open(temp_pdf_name, 'rb') as pdf_file:
            s3.upload_fileobj(pdf_file, s3_bucket, pdf_output_key)

        print(f"Uploaded PDF to S3: s3://{s3_bucket}/{pdf_output_key}")

    except Exception as e:
        print(f"Error saving images to PDF and uploading to S3: {str(e)}")

def extract_text_from_image(s3_bucket, image_key):
    """
    Extract text from an image stored in S3 using Textract.

    Parameters:
    - s3_bucket: S3 bucket containing the image.
    - image_key: Key of the image file in S3.

    Returns:
    Extracted text from the image.
    """
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

def image_to_text(s3_bucket, image_key):
    """
    Extract text content from an image stored in an S3 bucket using Amazon Textract.

    Parameters:
    - s3_bucket: The name of the S3 bucket.
    - image_key: The key (path) of the image file in the S3 bucket.

    Returns:
    - text_data: Extracted text content from the image.
    """
    # Start a Textract job to detect text in the specified image
    response = textract_client.start_document_text_detection(
        DocumentLocation={'S3Object': {'Bucket': s3_bucket, 'Name': image_key}}
    )
    job_id = response['JobId']  # Extract the JobId from the Textract response

    # Wait for the Textract job to complete
    while True:
        job_response = textract_client.get_document_text_detection(JobId=job_id)
        status = job_response['JobStatus']

        if status in ['SUCCEEDED', 'FAILED']:
            break
        elif status == 'IN_PROGRESS':
            import time
            time.sleep(5)  # Wait for 5 seconds before checking the job status again
        else:
            raise Exception(f"Textract job failed with status: {status}")

    text_data = ''
    # Extract text content from Textract response blocks
    for item in job_response['Blocks']:
        if item['BlockType'] == 'LINE':
            text_data += item['Text'] + '\n'

    return text_data  # Return the extracted text content from the image

def parse_date(date_str):
    """
    Parse a date string using multiple date formats and return the parsed date in a standardized format.

    Parameters:
    - date_str: A string representing a date.

    Returns:
    - formatted_date: The parsed date in the format 'YYYY-MM-DD' or None if parsing fails.
    """
    formats = ["%m/%d/%Y", "%Y-%m-%d", "%B %d, %Y","%B %d,%Y"]  # List of accepted date formats

    if date_str:  # Check if the input date string is not empty
        for date_format in formats:
            try:
                # Attempt to parse the date using the current format
                parsed_date = datetime.strptime(date_str, date_format)
                return parsed_date.strftime("%Y-%m-%d")  # Return the formatted date
            except ValueError:
                pass  # Continue to the next format if the current one fails
    
    return None  # Return None if the date string cannot be parsed using any of the specified formats

def extract_totalpayment_duedate_companyname(s3_bucket, image_key,df):
    """
    Extracts due date and total payment information from an image stored in an S3 bucket.

    Parameters:
    - s3_bucket: The S3 bucket name where the image is stored.
    - image_key: The key (path) of the image file within the S3 bucket.

    Returns:
    - (due_date, total_payment): A tuple containing the due date (formatted) and the total payment amount.
    """
    # Extract text content from the image using Textract
    text = image_to_text(s3_bucket, image_key)
    invoice_info =name_invoice(text,df)

    pdf_name = invoice_info['invoice_name']
    due_date = parse_date(invoice_info['invoice_name'])
    total_payment = invoice_info['total_amount']
    company_name= invoice_info['company_name']

    if company_name is not None:
        companyname=company_name
    else:
        companyname='not exist'
    # Return a tuple containing the due date and total payment
    return pdf_name, due_date, total_payment, companyname

def name_invoice(text,df):
    """
    Extracts information from the text, including customer name, due date, and total amount.
    Renames the S3 file based on the extracted information.

    Parameters:
    - text (str): The text from which information needs to be extracted.
  
    Returns:
    dict: A dictionary containing extracted information and the renamed PDF file name.
    """

    # Extract customer name, due date, and total amount from the text
    customer_name = extract_customer_name(text)
    _,date = extract_date(text)
    total_amount = extract_amount(text)
    company_name=extract_company_name(df, text)

    invoice_name = None  

    #------------------------------------------
    # Check all possible cases
    if date is not None and customer_name is not None and total_amount is not None and company_name is not None:
        # Sanitize customer name by replacing special characters with underscores
        sanitized_customer_name = re.sub(r'[\\/:"*?<>|]', '', customer_name)
        # Create a destination name with date, formatted total amount, and sanitized customer name
        invoice_name = f"{date} {format_currency(total_amount)} {company_name} {sanitized_customer_name}.pdf"
    
    elif date is None and customer_name is not None and total_amount is not None and company_name is not None:
        sanitized_customer_name = re.sub(r'[\\/:"*?<>|]', '', customer_name)
        invoice_name = f"{format_currency(total_amount)} {company_name} {sanitized_customer_name}.pdf"
   
    elif date is not None and customer_name is None and total_amount is not None and company_name is not None:
        
        invoice_name = f"{date} {format_currency(total_amount)} {company_name}.pdf"
   
    elif date is not None and customer_name is not None and total_amount is None and company_name is not None:
        sanitized_customer_name = re.sub(r'[\\/:"*?<>|]', '', customer_name)
        invoice_name = f"{date} {company_name} {sanitized_customer_name}.pdf"
   
    elif date is not None and customer_name is not None and total_amount is not None and company_name is None:
        sanitized_customer_name=re.sub(r'[\\/:"*?<>|]', '', customer_name)
        invoice_name = f"{date} {format_currency(total_amount)} {sanitized_customer_name}.pdf"
   
    elif date is None and customer_name is None and total_amount is not None and company_name is not None:
        
        invoice_name = f"{format_currency(total_amount)} {company_name}.pdf"
   
    elif date is None and customer_name is not None and total_amount is None and company_name is not None:
        sanitized_customer_name=re.sub(r'[\\/:"*?<>|]', '', customer_name)
        invoice_name = f"{company_name} {sanitized_customer_name}.pdf"

    elif date is not None and customer_name is  None and total_amount is not None and company_name is  None:
        invoice_name = f"{date} {format_currency(total_amount)}.pdf"
    
    elif total_amount is not None and company_name is  None and date is  None and customer_name is  None:
        
        invoice_name = f"{format_currency(total_amount)}.pdf"

    elif total_amount is not None and company_name is  None and date is  None and customer_name is  None:
        
        invoice_name = f"{total_amount}.pdf"



    # Return a dictionary containing extracted information and the renamed PDF file name
    return {
        'invoice_name': invoice_name,
        'Due Date': date,
        'Customer Name': customer_name,
        'company_name': company_name,
        'total_amount':total_amount,
    }

def splitpdf_into_invoices(s3_bucket, file_content_array, images_keys_np, split_prefix,df):
    """
    Processes a set of image pages, extracts relevant information, and organizes similar pages into PDFs.

    Parameters:
    - s3_bucket: The S3 bucket where the images are stored.
    - file_content_array: Array containing binary image data.
    - images_keys_np: Array containing keys of images in the S3 bucket.
    - split_prefix: Prefix for splitting the images.

    Returns:
    None
    """
    try:
        # Initialize variables for reference information and tracking similar pages
        reference_roi_gray = None
        reference_due_date = None
        reference_totalpayment = None
        reference_companyname = None


        similar_pages = []
        pdf_index = 1
        temp_pdf_name = None
        reference_pdf_name="unknow"

        # Iterate over each image content in the array
        for i, file_content in enumerate(file_content_array, start=1):
            # Decode image content and resize if necessary
            img_np = cv.imdecode(np.fromstring(file_content, np.uint8), cv.IMREAD_COLOR)
            max_display_height = 800
            max_display_width = 1200
            height, width = img_np.shape[:2]
            if height > max_display_height or width > max_display_width:
                scale_factor = min(max_display_height / height, max_display_width / width)
                img = cv.resize(img_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA)

            # Define the region of interest (ROI) and apply denoising
            start_x, end_x, start_y, end_y = 0, 250, 0, 70
            roi = img_np[start_y:end_y, start_x:end_x]
            img = cv.fastNlMeansDenoisingColored(roi, None, 10, 10, 7, 21)
            roi_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            temp_pdf_name = f"/tmp/pdf_{pdf_index}.pdf"
            
            current_pdf_name,current_due_date, current_totalpayment,current_company_name = extract_totalpayment_duedate_companyname(s3_bucket, images_keys_np[i-1],df)
            # Extract account number and total payment from the current page
            if reference_roi_gray is not None:
               
                print(f'page {i}  compnay name --->{current_company_name}')
                # Compare the current ROI with the reference ROI using SSIM
                similarity_index = ssim(reference_roi_gray, roi_gray)
                print(f" {current_company_name},previous cp :{reference_companyname} ===>Similarity Index between Page {i} and Page {i-1}: {similarity_index}")

                # Set a similarity threshold
                similarity_threshold = 0.90

                # Check if the page is similar to the previous one based on SSIM or extracted information
                if similarity_index >= similarity_threshold and  (current_company_name==reference_companyname) :  # or (current_account_number == reference_account_number and current_totalpayment == reference_totalpayment)
                    print(f"Page {i} is similar to the previous page. Adding to the current sequence.")
                    print(f"{current_company_name},previous  :{reference_companyname} for Page {i} and previous {i-1}")
                    similar_pages.append(img_np)
                # Check if the page is similar to the previous one based on SSIM or extracted information
                elif current_company_name==reference_companyname :  # or (current_account_number == reference_account_number and current_totalpayment == reference_totalpayment)
                    print(f"Page {i} is similar to the previous page. Adding to the current sequence.")
                    print(f"{current_company_name},previous  :{reference_companyname} for Page {i} and previous {i-1}")
                    similar_pages.append(img_np)
                else:
                    # Pages are not similar, save the current sequence to a new PDF
                    if similar_pages:
                        print("Pages are not similar. Saving similar pages.")
                        save_images_to_pdf_and_upload(similar_pages, s3_bucket, split_prefix, reference_pdf_name, temp_pdf_name)
                        print(f"Saved similar pages to {temp_pdf_name}")
                        print(f"{current_company_name},previous cp :{reference_companyname} for Page {i} and previous {i-1}")

                    pdf_index += 1
                    # Start a new sequence with the current page
                    similar_pages = [img_np]
            else:
                print(f'page {i}  compnay name --->{current_company_name}')
                similar_pages.append(img_np)

            # Update reference information with the current page
            reference_roi_gray = roi_gray
            reference_companyname = current_company_name
            reference_pdf_name=current_pdf_name
            reference_due_date= current_due_date
            reference_totalpayment = current_totalpayment
            

        # Save the last sequence to a new PDF if not empty
        if similar_pages:
            # # Generate PDF names and paths
            # pdf_name = f"pdf_{pdf_index}.pdf"
            temp_pdf_name = f"/tmp/pdf_{pdf_index}.pdf"
            print("Save the last sequence to a new PDF if not empty.")
            save_images_to_pdf_and_upload(similar_pages, s3_bucket, split_prefix, reference_pdf_name, temp_pdf_name)
            print(f"Saved the last pages to {temp_pdf_name}")

    except Exception as e:
        print(f"Error processing pages: {str(e)}")

def start_textract_job(object_key, s3_bucket):
    """
    Initiates a Textract job for text detection on the specified S3 object.

    Parameters:
    - object_key: The key of the S3 object to process.
    - s3_bucket: The S3 bucket containing the object.

    Returns:
    str: The JobId associated with the initiated Textract job.
    """
    # Start a Textract job for text detection
    response = textract_client.start_document_text_detection(
        DocumentLocation={'S3Object': {'Bucket': s3_bucket, 'Name': object_key}}
    )
    return response['JobId']

def wait_for_textract_completion(job_id):
    """
    Waits for the completion of a Textract job with the specified JobId.

    Parameters:
    - job_id: The JobId of the Textract job to monitor.

    Returns:
    dict: The Textract response containing job status and results.
    """
    try:
        while True:
            # Check the status of the Textract job
            response = textract_client.get_document_text_detection(JobId=job_id)
            status = response['JobStatus']

            # Handle different job statuses
            if status in ['SUCCEEDED', 'FAILED', 'PARTIAL_SUCCESS']:
                return response
            elif status == 'IN_PROGRESS':
                time.sleep(5)  # Adjust the sleep duration as needed
            else:
                raise ValueError(f"Unexpected job status: {status}")

    except Exception as e:
        print(f"Error in wait_for_textract_completion: {str(e)}")

def pdf_to_text(pdf_key, s3_bucket):
    """
    Converts a PDF file to text using Amazon Textract.

    Parameters:
    - pdf_key: The key of the PDF file in the S3 bucket.
    - s3_bucket: The S3 bucket containing the PDF file.

    Returns:
    str or None: The extracted text from the PDF or None if an error occurs.
    """
    # Start a Textract job for text detection on the PDF
    job_id = start_textract_job(pdf_key, s3_bucket)

    try:
        # Wait for the Textract job to complete
        wait_for_textract_completion(job_id)

        # Get Textract results for the completed job
        results = textract_client.get_document_text_detection(JobId=job_id)

        # Extract text lines from the Textract results
        text = "\n".join(result['Text'] for result in results['Blocks'] if result['BlockType'] == 'LINE')

        return text
    except Exception as e:
        print(f"Error extracting content from PDF {pdf_key}: {str(e)}")

    return None

def rename_s3_file(bucket_name, old_key, new_key):
    """
    Renames a file in an S3 bucket by copying it to a new key and deleting the old key.

    Parameters:
    - bucket_name: The name of the S3 bucket.
    - old_key: The current key of the file to be renamed.
    - new_key: The new key for the file.

    Returns:
    None
    """
    # Copy the object to the new key
    s3.copy_object(Bucket=bucket_name, CopySource={'Bucket': bucket_name, 'Key': old_key}, Key=new_key)
    
    # Delete the object with the old key
    s3.delete_object(Bucket=bucket_name, Key=old_key)

def process_single_image(image_key, s3_bucket, conditions):
    """
    Processes a single image stored in an S3 bucket using Amazon Textract.

    Parameters:
    - image_key: The key of the image file in the S3 bucket.
    - s3_bucket: The S3 bucket containing the image file.
    - conditions: A list of conditions to check in the extracted text.

    Returns:
    str or None: The image key if the specified conditions are met, or None otherwise.
    """
    # Start a Textract job for text detection on the image
    job_id = start_textract_job(image_key, s3_bucket)

    try:
        # Wait for the Textract job to complete
        wait_for_textract_completion(job_id)

        # Get Textract results for the completed job
        results = textract_client.get_document_text_detection(JobId=job_id)
        
        # Extract text lines from the Textract results
        text = "\n".join(result['Text'] for result in results['Blocks'] if result['BlockType'] == 'LINE')

        # Check if any condition is met and 'Dear' is not in the text
        if any(condition in text for condition in conditions) and ('Dear' or 'S M T W T F S') not in text:
            return image_key

    except Exception as e:
        print(f"Error processing single image {image_key}: {str(e)}")

    return None



def convert_date(input_date):
    
    """
    Converts the input date string into a standardized output date format.

    Parameters:
    - input_date (str): The input date string to be converted.
    """
    # Convert string to datetime object
    date_object = datetime.strptime(input_date, "%Y-%m-%d")

    # Format the datetime object as ymd (year-month-day without separators)
    output_format = date_object.strftime("%y%m%d")
    return output_format

def extract_date(text):
    """
    Extracts the due date from the given text using a predefined pattern.

    Parameters:d
    - text (str): The text from which the due date needs to be extracted.

    Returns:
    str: The extracted due date in the format "%y%m%d" or None if not found.
    """
    formatted_date=None

    date_pattern = re.compile(r'\b(?:\d{1,2}[/]\d{1,2}[/]\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\s\d{1,2},?\s?\d{2,4})\b')
    # Find all matches in the text
    date_match = date_pattern.findall(text.lower())

    # # Parse the found dates using dateutil.parser
    parsed_dates = [parser.parse(match) for match in date_match]

    # formatted_date_convert=formatted_date
    # If a date is found, print it
    if parsed_dates:
        date=parsed_dates[0].strftime("%Y-%m-%d")
        formatted_date=date
        formatted_date_convert = convert_date(date) if date else None
        
    
    return formatted_date,formatted_date_convert


def findelement(customer_name_match):
    """
    Extracts and cleans the customer name from the given match object.

    Parameters:
    - customer_name_match (re.Match): The match object containing customer name information.

    Returns:
    str: The extracted and cleaned customer name.
    """
    # Extract the raw text from the match object and replace newline characters with spaces
    text = customer_name_match.group(1).replace('\n', ' ')

    # Find the index of the colon character in the text
    colon_index = text.find(':')

    # If a colon is found, extract the text after the colon (ignoring leading and trailing spaces)
    if colon_index != -1:
        result = text[colon_index + 1:].strip().strip("'")
    else:
        # If no colon is found, use the entire text
        result = text

    return result

def format_currency(amount):
    """
    Formats the given amount as a currency string with grouping.

    Parameters:
    - amount (float): The amount to be formatted.

    Returns:
    str: The formatted currency string.
    """
    # Set the locale to the user's default for currency formatting
    locale.setlocale(locale.LC_ALL, '')
    
    # Format the amount as a currency string with grouping
    formatted_currency = locale.currency(amount, grouping=True)
    
    return formatted_currency

# Function to convert string to float based on condition
def convert_to_float(amount_str):
    if amount_str.startswith('$'):
        amount_str = amount_str[1:]  # Remove the dollar sign
    amount_str = amount_str.replace(',', '')  # Remove commas
    return float(amount_str)

def extract_amount(text):
    """
    Extracts the total amount due from the given text using a predefined pattern.

    Parameters:
    - text (str): The text from which the total amount needs to be extracted.

    Returns:
    float: The extracted total amount as a float.
    """
    currency_pattern = re.compile(r'\$(\s*[0-9,]+(?:\.[0-9]{2})?)')

    # Search for the total amount in the text
    amount_match = currency_pattern.search(text)

     # If an amount is found, print it
    if amount_match:
        extracted_amount = amount_match.group(1)
        # print(f"Total amount:", extracted_amount)
        return  convert_to_float(extracted_amount)

    # Return None if no match is found
    return None

def correct_customer_name(sentence):
    word=sentence.split('\n')
    if len(word)>1:
        w=word[0].replace(' ','')
        
        if len(w)<=5 and len(word[1])>5:
            result  = word[1]
                   
        else:
            result  = word[0]
    else:
         result  = word[0]
    return result

def extract_customer_name(text):

    customer_name_match1 = re.search(r'(?<=is billing\s)(.*?)(?=\sfor the month)|billed to:?\n?(.*)|from:\n?(.*)|bill To:?\n?(.*)|bill to\n?(.*)|site Name:?\n?(.*)|(.*)\npo box 853|(.+?)\n(.+?)\n(.+?)p\.o\. box 853|((?:.*\n){3})(po|p.o|p.o.)\s*(box|box) 853|client name:?\s*(.*)',text.lower())

    if customer_name_match1 is not None :
        x=next((x for x in customer_name_match1.groups() if x is not None), "default") 
        # if customer_name_match1 and customer_name_match1.group(1).strip()!= "":
        # print(f'customer_name before:{x}')
        customer_name = ' '.join(re.findall(r'\b[A-Z][A-Z\s]+\b', x))

        if len(customer_name)>3:
            c=  customer_name
        else:
            c=  x
        return  correct_customer_name(c)
            # return findelement(potential_names)
                    
    else:
        return None

def check_company_name(company, text_to_search):
    if ' '+company.lower()+' ' in text_to_search.lower():
            return company
    elif ''+company.lower()+' ' in text_to_search.lower():
   
        return company
    elif ''+company.lower()+'/' in text_to_search.lower():
   
        return company
    else:
        return None

def extract_company_name(df, text):
    
    # Extract the company names from the DataFrame
    company_names = df['Company Name'].tolist()
    matching_company = None

    # Use multithreading to speed up the process
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(check_company_name, company, text.replace('\n', ' ')): company for company in company_names}
        for future in as_completed(futures):
            company_result = future.result()
            if company_result:
                matching_company=company_result
    return matching_company


def check_s3_object_exists(bucket_name, object_key):
    """
    Checks if an object exists in an S3 bucket.

    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - object_key (str): The key or name of the object in the S3 bucket.

    Returns:
    bool: True if the object exists, False otherwise.
    """
    try:
        # Attempt to retrieve the object metadata; if successful, the object exists
        s3.head_object(Bucket=bucket_name, Key=object_key)
        return True  # Object exists
    except Exception as e:
        if e.response['Error']['Code'] == '404':
            return False  # Object does not exist
        else:
            raise

def write_content_to_file(file_path, content):
    """
    Writes content to a file.

    Parameters:
    - file_path (str): The path to the file.
    - content (str): The content to be written to the file.
    """
    with open(file_path, 'w') as file:
        file.write(content)

def upload_to_s3(bucket, key, data, content_type):
    """
    Uploads data to an S3 bucket.

    Parameters:
    - bucket (str): The name of the S3 bucket.
    - key (str): The key or name of the object in the S3 bucket.
    - data (bytes or str): The data to be uploaded.
    - content_type (str): The content type of the data.

    Note: If data is a string, it will be encoded to bytes before uploading.
    """
    try:
        # Upload the data to the specified S3 bucket and key with the given content type
        s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")

def extract_invoice_info( pdf_file, text,df):
    """
    Extracts information from a given text based on predefined patterns.

    Parameters:
    - s3_bucket (str): The name of the S3 bucket.
    - pdf_file (str): The name of the PDF file.
    - text (str): The text content to extract information from.

    Returns:
    dict: A dictionary containing extracted information with field names as keys.
    """
    # Define patterns for extracting specific fields from the text
    patterns = {
        "Customer/client": None,
        "Due Date": None,
        "Total amount": None,
        "company name":None
    }
    
    # Initialize a dictionary to store extracted information
    extracted_info = {
        "Link to PDF": f'https://invoicesgranvilleinternational.s3.amazonaws.com/{pdf_file}'
        
    }

    # Loop through predefined patterns and extract information
    for field, pattern in patterns.items():
        
        customer_name = extract_customer_name(text)
        # date = extract_due_date(text)
        total_amount = extract_amount(text)
        company_name=extract_company_name(df, text)
        duedate,_=extract_date(text)

        if field=='Customer/client':
            extracted_info[field] = customer_name
        elif field=='company name':
            extracted_info[field] = company_name  
        elif field=="Total amount":
           extracted_info[field] = total_amount
        else:
           extracted_info[field] = duedate

    return extracted_info

def download_file(bucket, key, local_path):
    """
    Downloads a file from an S3 bucket and saves it to a local path.

    Parameters:
    - bucket (str): The name of the S3 bucket.
    - key (str): The key of the object (file) in the S3 bucket.
    - local_path (str): The local path where the file will be saved.
    """
    try:
        s3.download_file(bucket, key, local_path)
    except Exception as e:
        print(f"Error downloading file {key} from bucket {bucket}: {str(e)}")

def convert_page_to_image(pdf_path, page_number):
    """
    Converts a specific page of a PDF file to a JPEG image with enhanced quality.

    Parameters:
    - pdf_path (str): The path to the PDF file.
    - page_number (int): The page number to convert.

    Returns:
    BytesIO: A BytesIO buffer containing the converted JPEG image.
    """
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
        print(f"Error converting page {page_number} to image: {str(e)}")

def pdf_to_images(pdf_path, predix, s3_bucket):
    """
    Converts each page of a PDF file to JPEG images and uploads them to an S3 bucket.

    Parameters:
    - pdf_path (str): The path to the PDF file.
    - predix (str): Prefix to be used for naming the images in S3.
    - s3_bucket (str): The name of the S3 bucket.

    Returns:
    list: A list of image keys in the S3 bucket.
    """
    try:
        images = []
        # loop on all pages of the pdf and  convert them into images
        for i in range(1, len(fitz.open(pdf_path)) + 1):
            image_buffer = convert_page_to_image(pdf_path, i)
            image_key = f'{predix}/page_{i}.jpg'
            upload_to_s3(s3_bucket, image_key, image_buffer.getvalue(), 'image/jpeg')
            # add image key in the list
            images.append(image_key)
        return images

    except Exception as e:
        print(f"Error converting PDF to images: {str(e)}")

#  Extract Features and save them into csv file then ulpoad it in s3
def saveextractFeatures_savethem_into_csv(s3_bucket, pdf_file, output_bucket, all_csv_data, header_written,outputcsvname,df):
    """
    Processes a single PDF file, extracts information, and saves data to a CSV file.

    Parameters:
    - s3_bucket (str): The S3 bucket containing the PDF file.
    - pdf_file (str): The key of the PDF file in the S3 bucket.
    - output_bucket (str): The S3 bucket to save the CSV file.
    - all_csv_data (list): List containing all rows of CSV data.
    - header_written (bool): Flag indicating whether the CSV header has been written.

    Returns:
    None
    """
    text_data = pdf_to_text(pdf_file, s3_bucket)
    extracted_info = extract_invoice_info( pdf_file, text_data,df)
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
        Key=outputcsvname,
        Body=csv_buffer.getvalue()
    )

# read company names from csv files
def readcompnaynamefile(invoices_s3_bucket,file_key):
    # Download CSV file to /tmp/
    s3.download_file(invoices_s3_bucket, file_key, f'/tmp/{file_key}')
    companiesdf = pd.read_csv(f'/tmp/{file_key}')
    return companiesdf


# this function will call the saveextractFeatures_savethem_into_csv function for iterate through each invoice PDF in the SaveInvoices folder
def process_invoicepdfs_to_csv(invoices_s3_bucket,SaveInvoices_prefix,output_bucket,outputcsvname,df):
    # List split PDFs in the SaveInvoices folder
    response = s3.list_objects_v2(Bucket=invoices_s3_bucket, Prefix=SaveInvoices_prefix)

    all_csv_data = []  # Initialize an empty list to accumulate CSV data
    header_written = False

    # Iterate through each split PDF in the SaveInvoices folder and process data
    for i, obj in enumerate(response.get('Contents', [])):
        pdf_file = obj['Key']
        if pdf_file.lower().endswith('.pdf'):
            # print(f'Link to PDF: {pdf_file}')
            saveextractFeatures_savethem_into_csv(invoices_s3_bucket, pdf_file, output_bucket, all_csv_data, header_written,outputcsvname,df)
            header_written = True 

# this function will convert pdf file into images
def ConvertPDFpages_to_images_and_sort(invoices_s3_bucket,images_folder,pdf_file):
    local_pdf_path = f'/tmp/{pdf_file}'
    download_file(input_s3_bucket, pdf_file, local_pdf_path)
    # Convert PDF pages to images and extract relevant information
    sorted_images = sorted(pdf_to_images(local_pdf_path, images_folder, invoices_s3_bucket), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return sorted_images



def splitInvoiceToInvoicespdf_savefeaturesextractintocsv(invoices_s3_bucket,extracted_images,SaveInvoices_prefix,outputcsvname,df):

    if extracted_images:
        image_contents = []
        images_keys_np = []
        for image_key in extracted_images:
            local_image_path = f"/tmp/{os.path.basename(image_key)}"
            download_file(invoices_s3_bucket, image_key, local_image_path)
            with open(local_image_path, 'rb') as image_file:
                image_contents.append(image_file.read())
            images_keys_np.append(image_key)

        if image_contents:

            # split scanned pdf documents into invoices based on name the invoices and images' similarity
            splitpdf_into_invoices(invoices_s3_bucket, image_contents, images_keys_np, SaveInvoices_prefix,df)
        
        #  
        process_invoicepdfs_to_csv(invoices_s3_bucket,SaveInvoices_prefix,output_bucket,outputcsvname,df)


def handler(event, context):
       
    df=readcompnaynamefile(invoices_s3_bucket,file_key)
    # Retrieve a list of objects in the input S3 bucket
    objects = s3.list_objects(Bucket=input_s3_bucket)
    # Iterate through each object in the input S3 bucket
    for obj in objects.get('Contents', []):
        pdf_file = obj['Key']

        # Check if the object is a PDF file
        if pdf_file.lower().endswith('.pdf'):
            extracted_images=ConvertPDFpages_to_images_and_sort(invoices_s3_bucket,images_folder,pdf_file)
            # Process the extracted images and create split PDFs and process invoice pdfs to create csv file from invoice pdfs
            splitInvoiceToInvoicespdf_savefeaturesextractintocsv(invoices_s3_bucket,extracted_images,"{}/{}".format(SaveInvoices_prefix, pdf_file.replace(".pdf", "")),pdf_file.replace("pdf", "csv"),df)
            
                
    return {
        'statusCode': 200,
        'body': 'Textract jobs completed for all PDFs in the input bucket.'
    }
    