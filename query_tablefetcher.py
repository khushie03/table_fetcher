import os
import csv
from fuzzywuzzy import process

def find_best_match(query, choices):
    """Find the best match from choices based on the query using fuzzy matching."""
    best_match, score = process.extractOne(query, choices)
    return best_match, score

def extract_first_row_from_csv(csv_file_path):
    """Extract the first row from a CSV file."""
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        first_row = next(reader, [])
    return first_row

def process_extracted_tables(csv_dir, query):
    """Process all extracted tables to match their first row with the query."""
    matches = []
    for csv_file in os.listdir(csv_dir):
        if csv_file.endswith(".csv"):
            base_name = os.path.splitext(csv_file)[0]  
            csv_file_path = os.path.join(csv_dir, csv_file)
            first_row = extract_first_row_from_csv(csv_file_path)
            first_row_text = " ".join(first_row)  
            best_match, score = find_best_match(query, [first_row_text])
            matches.append((base_name, best_match, score, csv_file_path))

    matches.sort(key=lambda x: x[2], reverse=True)
    
    return matches

def write_best_match_contents_to_csv(best_match_csv_path, output_file):
    """Write the contents of the best match CSV file to the output CSV file."""
    with open(best_match_csv_path, newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            writer.writerow(row)

pdf_dir = "C:/PROJECTS/pdf_layout_analysis/"
csv_dir = "C:/PROJECTS/pdf_layout_analysis/extracted_tables"
query = "ASSET WEALTH"
output_file = "C:/PROJECTS/pdf_layout_analysis/output.csv"

matches = process_extracted_tables(csv_dir, query)

for base_name, match, score, file_path in matches:
    print(f"Base Name: {base_name}")
    print(f"Matched Text: {match}")
    print(f"Score: {score}")
    print(f"File Path: {file_path}")
    print()

if matches:
    best_match_details = matches[0]
    best_match_csv_path = best_match_details[3]
    
    write_best_match_contents_to_csv(best_match_csv_path, output_file)
    print(f"Contents of the best match CSV file written to {output_file}")
else:
    print("No matches found.")
