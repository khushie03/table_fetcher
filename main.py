import fitz
import os
import requests
import csv
from PIL import Image
from paddleocr import PaddleOCR

API_URL = "https://api-inference.huggingface.co/models/microsoft/table-transformer-detection"
headers = {"Authorization": "Bearer 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"}

ocr_model = PaddleOCR(lang='en')

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def save_table_from_image(image_path, output_image_dir, box, table_count):
    with Image.open(image_path) as img:
        cropped_img = img.crop((box['xmin'], box['ymin'], box['xmax'], box['ymax']))
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        table_img_path = os.path.join(output_image_dir, f"{name}_table_{table_count}{ext}")
        cropped_img.save(table_img_path)
        return table_img_path

def get_y_coordinate(box):
    return box[0][1]

def process_table_image(image_path, output_csv_dir):
    result = ocr_model.ocr(image_path)
    
    flattened_result = [item for sublist in result for item in sublist]

    grouped_data = []
    current_group = []
    previous_y = None

    for item in flattened_result:
        box = item[0]
        text, confidence = item[1]
        
        y = get_y_coordinate(box)
        if previous_y is None or abs(y - previous_y) < 15:
            current_group.append(text)
        else:
            grouped_data.append(current_group)
            current_group = [text]
        previous_y = y

    if current_group:
        grouped_data.append(current_group)

    csv_file = os.path.join(output_csv_dir, f"{os.path.basename(image_path).replace('.png', '_ocr_output.csv')}")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in grouped_data:
            writer.writerow(row)

    print(f'OCR data for {image_path} has been written to {csv_file}')
    
    clean_csv_file(csv_file)

def clean_csv_file(csv_file):
    temp_file = csv_file + ".tmp"
    
    with open(csv_file, 'r', newline='') as infile, open(temp_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            cleaned_row = [cell for cell in row if '$' not in cell]
            writer.writerow(cleaned_row)
    
    os.replace(temp_file, csv_file)
    print(f'Cleaned CSV file: {csv_file}')

def extract_tables_from_pdf(pdf_path, output_image_dir="C:\\PROJECTS\\pdf layout\\images_output", output_csv_dir="C:\\PROJECTS\\pdf layout\\output_table", dpi=300, confidence_threshold=0.96):
    pdf_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    image_dir = os.path.join(output_image_dir, "pages")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    magnify = fitz.Matrix(zoom, zoom)

    for count, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=magnify)
        image_path = os.path.join(image_dir, f"{pdf_base_name}_page_{count}.png")
        pix.save(image_path)

    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        output = query(image_path)
        
        if output:
            table_count = 1
            for detection in output:
                if 'box' in detection and detection.get('score', 0) > confidence_threshold:
                    box = detection['box']
                    table_img_path = save_table_from_image(image_path, output_image_dir, box, table_count)
                    
                    process_table_image(table_img_path, output_csv_dir)
                    
                    table_count += 1
        else:
            print(f"No tables detected in {image_file}.")

pdf_path = r"C:\PROJECTS\pdf layout\accenture_report.pdf" 
extract_tables_from_pdf(pdf_path)
