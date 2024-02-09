# Import libraries 
import json
import boto3
import io
import cv2 as cv
from skimage.metrics import structural_similarity as ssim
import numpy as np
from fpdf import FPDF
import time
from datetime import datetime
import re
import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz
from io import BytesIO
from PIL import Image
import locale
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

    # Define a regular expression pattern to identify due dates
    due_date_pattern = re.compile(r'(Cigna.|\n|Bill Date:\s*|TOTAL AMOUNT DUE|Payment Due Date|Due Date|Invoice Date|DUE DATE)\s+((\d{1,2}/\d{1,2}/\d{2,4})|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{2,4})')

    # Search for the due date pattern in the extracted text
    due_date_match = due_date_pattern.search(text)

    # Extract the formatted date from the match or set to None if not found
    formatted_date = due_date_match.group(2) if due_date_match else None

    # Parse the formatted date using the parse_date function
    due_date = parse_date(formatted_date)

    # Extract the total payment amount using the total_Payment_Due function
    total_payment = total_Payment_Due(text)
    company_name= extract_company_name(df, text)
    if company_name is not None:
        companyname=company_name
    else:
        companyname='not exist'
    # Return a tuple containing the due date and total payment
    return due_date, total_payment, companyname

def total_Payment_Due(text):
    """
    Extracts the total payment amount from a given text.

    Parameters:
    - text: The text containing information about the total payment.

    Returns:
    - total_payment: The extracted total payment amount as a string or 'None' if not found.
    """
    # Define a regular expression pattern to identify total payment amounts
    totalpayment_pattern = re.compile(r'TOTAL AMOUNT DUE\s+\d{2}/\d{2}/\d{4}\s+\$([\d.]+)|Total Payment Due\n?.*?\n?([\d,]+\.\d+)|TOTAL DUE\s*\$([\d.]+)|AMOUNT DUE\s*([\d,.]+)|Total Due\s*\$([\d,.]+)|auto pay:?\s*\$([\d,.]+)|TOTAL AMOUNT DUE\s*.*?\n.*?\n\s*\$([\d.]+)')

    # Search for the total payment pattern in the given text
    totalpayment_match = totalpayment_pattern.search(text)

    # If a match is found, extract the first non-empty group, otherwise set to 'None'
    total_payment = next((x for x in totalpayment_match.groups() if x is not None), None) if totalpayment_match else None

    # Return the extracted total payment amount
    return total_payment

def process_pages(s3_bucket, file_content_array, images_keys_np, split_prefix,df):
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
        reference_account_number = None
        reference_totalpayment = None
        reference_companyname = None

        similar_pages = []
        pdf_index = 1
        temp_pdf_name = None

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

            # Generate PDF names and paths
            pdf_name = f"pdf_{pdf_index}.pdf"
            temp_pdf_name = f"/tmp/pdf_{pdf_index}.pdf"
            current_account_number, current_totalpayment,current_company_name = extract_totalpayment_duedate_companyname(s3_bucket, images_keys_np[i-1],df)
            # Extract account number and total payment from the current page
            if reference_roi_gray is not None:
               
                print(f'page {i}  compnay name --->{current_company_name}')
                # Compare the current ROI with the reference ROI using SSIM
                similarity_index = ssim(reference_roi_gray, roi_gray)
                print(f" {current_company_name},previous cp :{reference_companyname} ===>Similarity Index between Page {i} and Page {i-1}: {similarity_index}")

                # Set a similarity threshold
                similarity_threshold = 0.90

                # Check if the page is similar to the previous one based on SSIM or extracted information
                if similarity_index >= similarity_threshold or (current_company_name==reference_companyname) :  # or (current_account_number == reference_account_number and current_totalpayment == reference_totalpayment)
                    print(f"Page {i} is similar to the previous page. Adding to the current sequence.")
                    print(f"{current_company_name},previous cp :{reference_companyname} for Page {i} and previous {i-1}")
                    similar_pages.append(img_np)
                else:
                    # Pages are not similar, save the current sequence to a new PDF
                    if similar_pages:
                        print("Pages are not similar. Saving similar pages.")
                        save_images_to_pdf_and_upload(similar_pages, s3_bucket, split_prefix, pdf_name, temp_pdf_name)
                        print(f"Saved similar pages to {temp_pdf_name}")
                        print(f"{current_company_name},previous cp :{reference_companyname} for Page {i} and previous {i-1}")

                    pdf_index += 1
                    print(f'pdf_index:{pdf_index}')
                    # Start a new sequence with the current page
                    similar_pages = [img_np]
            else:
                print(f'page {i}  compnay name --->{current_company_name}')
                similar_pages.append(img_np)

            # Update reference information with the current page
            reference_roi_gray = roi_gray
            reference_companyname = current_company_name
            reference_account_number = current_account_number
            reference_totalpayment = current_totalpayment
            

        # Save the last sequence to a new PDF if not empty
        if similar_pages:
            # Generate PDF names and paths
            pdf_name = f"pdf_{pdf_index}.pdf"
            temp_pdf_name = f"/tmp/pdf_{pdf_index}.pdf"
            print("Save the last sequence to a new PDF if not empty.")
            save_images_to_pdf_and_upload(similar_pages, s3_bucket, split_prefix, pdf_name, temp_pdf_name)
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

def process_images_basedoncondition(images, s3_bucket):
    """
    Processes a list of images stored in an S3 bucket using Amazon Textract.

    Parameters:
    - images: A list of image keys in the S3 bucket.
    - s3_bucket: The S3 bucket containing the images.

    Returns:
    list: A list of image keys that meet specified conditions after processing with Textract.
    """
    try:
        # List of conditions to check in the extracted text
        conditions = ['Account Number', 'Customer Number', 'Loan Number', 'Customer ID', 'LOAN #']

        # List to store extracted images that meet the conditions
        extracted_images = []

        # Use ThreadPoolExecutor for concurrent processing of images
        with ThreadPoolExecutor() as executor:
            # Submit Textract processing tasks for each image in the list
            futures = [executor.submit(process_single_image, image_key, s3_bucket, conditions) for image_key in images]

            # Iterate over completed futures and collect results
            for future in as_completed(futures):
                result = future.result()
                if result:
                    extracted_images.append(result)

        return extracted_images

    except Exception as e:
        print(f"Error processing images: {str(e)}")

def convert_date(input_date):
    """
    Converts the input date string into a standardized output date format.

    Parameters:
    - input_date (str): The input date string to be converted.

    Returns:
    str: The converted date string in the format "%y%m%d".
    """
    supported_formats = ["%B %d, %Y", "%B %d,%Y", "%m/%d/%Y", "%m/%d/%y"]

    for format_str in supported_formats:
        try:
            parsed_date = datetime.strptime(input_date, format_str)
            output_date = parsed_date.strftime("%y%m%d")
            return output_date
        except ValueError:
            continue

    # If all parsing attempts fail, raise an error
    raise ValueError(f"time data {input_date!r} does not match any of the supported formats")

def extract_due_date(text):
    """
    Extracts the due date from the given text using a predefined pattern.

    Parameters:
    - text (str): The text from which the due date needs to be extracted.

    Returns:
    str: The extracted due date in the format "%y%m%d" or None if not found.
    """
    # Define a regular expression pattern to find due dates in the text
    due_date_pattern = re.compile(r'(Cigna.|\n|Bill Date:\s*|TOTAL AMOUNT DUE|Payment Due Date|Due Date|DUE DATE)\s+((\d{1,2}/\d{1,2}/\d{2,4})|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{2,4})')
    
    # Search for due dates using the defined pattern
    due_date_match = due_date_pattern.search(text)
    
    # Extract the second group from the match as the formatted date
    formatted_date = due_date_match.group(2) if due_date_match else None
    # print(f"formatted_date before:{formatted_date}")
    
    # If a formatted date is found, convert it to the standardized output format
    date = convert_date(formatted_date) if formatted_date else None

    return date

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

def extract_total_amount(text):
    """
    Extracts the total amount due from the given text using a predefined pattern.

    Parameters:
    - text (str): The text from which the total amount needs to be extracted.

    Returns:
    float: The extracted total amount as a float.
    """
    # Define a regular expression pattern to find total amounts in the text
    match = re.search(r'TOTAL AMOUNT DUE\s+\d{2}/\d{2}/\d{4}\s+\$([\d.]+)|Total Payment Due\n?.*?\n?([\d,]+\.\d+)|TOTAL DUE\s*\$([\d.]+)|AMOUNT DUE\s*([\d,.]+)|Total Due\s*\$([\d,.]+)', text)
    
    # If a match is found, extract the first non-empty group from the match
    if match:
        x = next((x for x in match.groups() if x is not None), "default")

        # If the amount contains commas, remove them before converting to float
        if ',' in x:
            amount_as_float = float(x.replace(',', ''))
        else:
            amount_as_float = float(x)

        return amount_as_float

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
        return  correct_customer_name(c)
            # return findelement(potential_names)
                    
    else:
        return None

def check_company_name(company, text_to_search):
    if ' '+company.lower()+' ' in text_to_search.lower():
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

def extract_information(s3_bucket, file_obj, SaveInvoicesprefix, text,df):
    """
    Extracts information from the text, including customer name, due date, and total amount.
    Renames the S3 file based on the extracted information.

    Parameters:
    - s3_bucket (str): The S3 bucket where the file is stored.
    - file_obj (str): The key or name of the file in the S3 bucket.
    - SaveInvoicesprefix (str): The prefix used for saving invoices.
    - text (str): The text from which information needs to be extracted.

    Returns:
    dict: A dictionary containing extracted information and the renamed PDF file name.
    """
    
    # Extract customer name, due date, and total amount from the text
    customer_name = extract_customer_name(text)
    date = extract_due_date(text)
    total_amount = extract_total_amount(text)
    company_name=extract_company_name(df, text)

    dest = None
    
    # Check if all required information is present
    if date and customer_name  and total_amount and company_name:
        # Sanitize customer name by replacing special characters with underscores
        sanitized_customer_name = re.sub(r'[\\/:"*?<>|]', '', customer_name)
        # Create a destination name with date, formatted total amount, and sanitized customer name
        dest = f"{SaveInvoicesprefix}/{date} {format_currency(total_amount)} {company_name} {sanitized_customer_name}.pdf"
    
    elif date and customer_name  and company_name:
        sanitized_customer_name = re.sub(r'[\\/:"*?<>|]', '', customer_name)
        dest = f"{SaveInvoicesprefix}/{date} {company_name} {sanitized_customer_name}.pdf"
    elif date and company_name:
        sanitized_customer_name = re.sub(r'[\\/:"*?<>|]', '', customer_name)
        dest = f"{SaveInvoicesprefix}/{date}{company_name}.pdf"
    elif date and customer_name:
        dest = f"{SaveInvoicesprefix}/{date}.pdf"
    else:
        dest=f"{file_obj}"


    # Rename the S3 file based on the extracted information
    rename_s3_file(s3_bucket, file_obj, dest)

    # Return a dictionary containing extracted information and the renamed PDF file name
    return {
        'renamepdf': dest,
        'Due Date': date,
        'Customer Name': customer_name,
        'company_name': company_name
    }

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

def extract_invoice_info(s3_bucket, pdf_file, text,df):
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
        "Customer/client": r'((?:.*\n){2})(PO|P.O|P.O.)\s*(BOX|Box) 853|(.*)(PO|P.O|P.O.)\s*(BOX|Box) 853',
        "Due Date": r'DUE DATE\n(\w{3} \d{1,2}, \d{4})|Payment Due Date\n(\d{2}/\d{2}/\d{4})|DUE DATE(\d{2}/\d{1,2}/\d{4})|Invoice Date\n(\d{2}/\d{1,2}/\d{4})|Bill Date: (\w+ \d{1,2},\s*\d{4})|Cigna.\n(\w+ \d{1,2},\s*\d{4})|DUE DATE\n(\d{2}/\d{1,2}/\d{2,4})',
        "Total amount": r'TOTAL AMOUNT DUE\s+\d{2}/\d{2}/\d{4}\s+\$([\d.]+)|Total Payment Due\n?.*?\n?([\d,]+\.\d+)|TOTAL DUE\s*\$([\d.]+)|AMOUNT DUE\s*([\d,.]+)|Total Due\s*\$([\d,.]+)',
        "company name": r'PAYABLE TO:\s+00110?\s*(\w+\s?\w+)',
        "Property Address/Service Address": r'Property Address:\n(.+)|Service Address\s+?Page?\s+?1?\s+?of?\s+?2?\n\w*\n\w*\n(\w{1,4}\s+\w*\s+\w*)',
    }
    
    # Initialize a dictionary to store extracted information
    extracted_info = {
        "Link to PDF": f's3://{s3_bucket}/{pdf_file}'
    }

    # Loop through predefined patterns and extract information
    for field, pattern in patterns.items():
        
        customer_name = extract_customer_name(text)
        # date = extract_due_date(text)
        total_amount = extract_total_amount(text)
        company_name=extract_company_name(df, text)
        match = re.search(pattern, text)
        if field=='Property Address/Service Address' or field== 'Due Date':
            
            if match:
                # Extract the matched group, choosing the first non-empty group
                x = next((x for x in match.groups() if x is not None), "default")  
                extracted_info[field] = x
            else:
                extracted_info[field] = '' 

        if field=='Customer/client':
            extracted_info[field] = customer_name
        elif field=='company name':
            extracted_info[field] = company_name  
        elif field=="Total amount":
           extracted_info[field] = total_amount

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

        for i in range(1, len(fitz.open(pdf_path)) + 1):
            image_buffer = convert_page_to_image(pdf_path, i)
            image_key = f'{predix}/page_{i}.jpg'
            upload_to_s3(s3_bucket, image_key, image_buffer.getvalue(), 'image/jpeg')
            images.append(image_key)
        return images

    except Exception as e:
        print(f"Error converting PDF to images: {str(e)}")

def process_single_pdf(s3_bucket, pdf_file, output_bucket, all_csv_data, header_written,outputcsvname,df):
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
    extracted_info = extract_invoice_info(s3_bucket, pdf_file, text_data,df)
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

def readcompnaynamefile(invoices_s3_bucket,file_key):
    # Download CSV file to /tmp/
    s3.download_file(invoices_s3_bucket, file_key, f'/tmp/{file_key}')
    companiesdf = pd.read_csv(f'/tmp/{file_key}')
    return companiesdf

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
            process_single_pdf(invoices_s3_bucket, pdf_file, output_bucket, all_csv_data, header_written,outputcsvname,df)
            header_written = True 

def fromsplitPDF_extract_information(invoices_s3_bucket,SaveInvoices_prefix,split_prefix,df):
    response = s3.list_objects_v2(Bucket=invoices_s3_bucket, Prefix=split_prefix)
    # Iterate through each split PDF and extract information
    for i, obj in enumerate(response.get('Contents', [])):
        file_obj = obj['Key']
        print(f'file_obj:{file_obj}')
        if file_obj and file_obj.lower().endswith('.pdf'):  # Check if file_obj is not None and is a PDF
            text_from_pdf = pdf_to_text(file_obj, invoices_s3_bucket)
            information = extract_information(invoices_s3_bucket, file_obj, SaveInvoices_prefix, text_from_pdf,df)
            print(f"Information extracted: {information}")

def ConvertPDFpages_to_images_and_sort(invoices_s3_bucket,images_folder,pdf_file):
    local_pdf_path = f'/tmp/{pdf_file}'
    download_file(input_s3_bucket, pdf_file, local_pdf_path)
    # print(f'Processing PDF: {local_pdf_path}')
    # Convert PDF pages to images and extract relevant information
    sorted_images = sorted(pdf_to_images(local_pdf_path, images_folder, invoices_s3_bucket), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    extracted_images = sorted(process_images_basedoncondition(sorted_images, invoices_s3_bucket), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return extracted_images

def Process_extracted_images_createsplitPDFs(invoices_s3_bucket,extracted_images,SaveInvoices_prefix,split_prefix,outputcsvname,df):

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
            process_pages(invoices_s3_bucket, image_contents, images_keys_np, split_prefix,df)
        
        fromsplitPDF_extract_information(invoices_s3_bucket,SaveInvoices_prefix,split_prefix,df)
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
            Process_extracted_images_createsplitPDFs(invoices_s3_bucket,extracted_images,"{}/{}".format(SaveInvoices_prefix, pdf_file.replace(".pdf", "")),"{}/{}".format(split_prefix, pdf_file.replace(".pdf", "")),pdf_file.replace("pdf", "csv"),df)
            
                

    return {
        'statusCode': 200,
        'body': 'Textract jobs completed for all PDFs in the input bucket.'
    }