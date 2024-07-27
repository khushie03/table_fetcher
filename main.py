import fitz
import os
from PIL import Image, ImageDraw
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection
from torchvision import transforms
import torch
import numpy as np
import csv
import easyocr
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def apply_ocr(cropped_table, cell_coordinates):
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []
        for cell in row["cells"]:
            cell_image = cropped_table.crop(cell["cell"])
            result = reader.readtext(np.array(cell_image))
            if len(result) > 0:
                text = " ".join([x[1] for x in result])
                row_text.append(text)
            else:
                print(f"No text detected in cell {cell['cell']}")

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[idx] = row_text

    print("Max number of columns:", max_num_columns)
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data

    return data

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
model.to(device)

pdf_path = "C:/PROJECTS/pdf_layout_analysis/financial_report.pdf"
output_dir = "./output_images"
os.makedirs(output_dir, exist_ok=True)

structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
structure_model.to(device)

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))
        return resized_image

def convert_pdf_to_images(pdf_path):
    dpi = 300
    zoom = dpi / 72
    magnify = fitz.Matrix(zoom, zoom)
    output_dir = "./images"
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for count, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=magnify)
        pix.save(os.path.join(output_dir, f"{base_name}_{count}.png"))

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

id2label = model.config.id2label
id2label[len(model.config.id2label)] = "no object"

def outputs_to_objects(outputs, img_size, id2label, score_threshold=0.97):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if class_label != 'no object' and score > score_threshold:
            objects.append({'label': class_label, 'score': float(score), 'bbox': [float(elem) for elem in bbox]})

    return objects

def plot_results(cells, class_to_visualize, cropped_table):
    if class_to_visualize not in structure_model.config.id2label.values():
        raise ValueError("Class should be one of the available classes")

    plt.figure(figsize=(16, 10))
    plt.imshow(cropped_table)
    ax = plt.gca()

    for cell in cells:
        score = cell["score"]
        bbox = cell["bbox"]
        label = cell["label"]

        if label == class_to_visualize:
            xmin, ymin, xmax, ymax = tuple(bbox)
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color="red", linewidth=3))
            text = f'{cell["label"]}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def objects_to_crops(img, tokens, objects, class_thresholds, padding=20, top_padding=200):  # Increased padding values
    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}
        bbox = obj['bbox']
        bbox = [bbox[0] - padding, bbox[1] - padding - top_padding, bbox[2] + padding, bbox[3] + padding]
        print(f"Detected table with bbox: {bbox} and score: {obj['score']}")  # Debugging print

        try:
            cropped_img = img.crop(bbox)
            table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
            for token in table_tokens:
                token['bbox'] = [token['bbox'][0] - bbox[0], token['bbox'][1] - bbox[1], token['bbox'][2] - bbox[0], token['bbox'][3] - bbox[1]]
            if obj['label'] == 'table rotated':
                cropped_img = cropped_img.rotate(270, expand=True)
                for token in table_tokens:
                    bbox = token['bbox']
                    bbox = [cropped_img.size[0] - bbox[3] - 1, bbox[0], cropped_img.size[0] - bbox[1] - 1, bbox[2]]
                    token['bbox'] = bbox

            cropped_table['image'] = cropped_img
            cropped_table['tokens'] = table_tokens
            table_crops.append(cropped_table)
        except Exception as e:
            print(f"Error cropping image: {e}")

    return table_crops

def get_cell_coordinates_by_row(table_data):
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})
        row_cells.sort(key=lambda x: x['column'][0])
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})
    cell_coordinates.sort(key=lambda x: x['row'][1])

    return cell_coordinates

def from_image(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    resized_image = image.resize((int(0.6 * width), int(0.6 * height)))
    detection_transform = transforms.Compose([
        MaxResize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pixel_values = detection_transform(resized_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(pixel_values)

    objects = outputs_to_objects(outputs, resized_image.size, id2label)

    tokens = []
    detection_class_thresholds = {
        "table": 0.97,
        "table rotated": 0.97,
        "no object": 10
    }
    crop_padding = 20

    tables_crops = objects_to_crops(resized_image, tokens, objects, detection_class_thresholds, padding=crop_padding, top_padding=90)

    for i, cropped_table in enumerate(tables_crops):
        output_path = os.path.join(output_dir, f"{base_name}_table_{i}.jpg")
        try:
            cropped_table['image'].save(output_path)
            print(f"Saved cropped table to {output_path}")

            structure_transform = transforms.Compose([
                MaxResize(1000),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            pixel_values = structure_transform(cropped_table['image']).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = structure_model(pixel_values)
            structure_id2label = structure_model.config.id2label
            structure_id2label[len(structure_id2label)] = "no object"
            cells = outputs_to_objects(outputs, cropped_table['image'].size, structure_id2label, score_threshold=0.90)
            print(cells)

            cropped_table_visualized = cropped_table['image'].copy()
            #draw = ImageDraw.Draw(cropped_table_visualized)

            #for cell in cells:
                #draw.rectangle(cell["bbox"], outline="red")
            #plot_results(cells, class_to_visualize="table row", cropped_table=cropped_table['image'])
            cell_coordinates = get_cell_coordinates_by_row(cells)
            data = apply_ocr(cropped_table['image'], cell_coordinates)
            for row, row_data in data.items():
                print(row_data)

            table_output_dir = "C:/PROJECTS/pdf_layout_analysis/extracted_tables"
            os.makedirs(table_output_dir, exist_ok=True)
            file_name = os.path.join(table_output_dir, f"{base_name}_table_{i}.csv")
            with open(file_name, 'w', newline='') as result_file:
                wr = csv.writer(result_file, dialect='excel')
                for row, row_text in data.items():
                    wr.writerow(row_text)
            print(f"Saved extracted table data to {file_name}")

        except Exception as e:
            print(f"Error processing table {i} in {image_path}: {e}")

convert_pdf_to_images(pdf_path)

image_files = [f for f in os.listdir("./images") if f.endswith(".png")]

for file_name in image_files:
    file_path = os.path.join("./images", file_name)
    from_image(file_path)
