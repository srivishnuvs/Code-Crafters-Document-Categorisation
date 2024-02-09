import pandas as pd
import shutil
import os

# Define the priority for each category (1 is highest priority)
category_priority = {
    'Medical': 1,
    'Legal': 2,
    'Finance': 3,
    'Business': 4,
    'Technological': 5,
    'Scientific': 6,
    'News articles': 7,
    'Government': 8,
    'Educational': 9,
    'Creative': 10
}

# Sort categories by priority
sorted_categories = sorted(category_priority, key=category_priority.get)

# Path to the CSV file
csv_file_path = 'C:\\Users\\Sri Vishnu V S\\Desktop\\your_csv_file.csv'  # Update this path

# Path to the destination folder
destination_folder_path = 'C:\\Users\\Sri Vishnu V S\\Desktop\\destination_folder'  # Update this path

# Keep track of processed PDFs
processed_pdfs = set()

# Function to process the CSV and move PDFs
def process_csv():
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Iterate over the categories based on the priority
    for category in sorted_categories:
        # Find rows where the category column has a value of 1 and the PDF has not been processed
        category_rows = df[(df[category] == 1) & (~df['PDF Path'].isin(processed_pdfs))]
        
        # Iterate over the rows and move the PDF files
        for _, row in category_rows.iterrows():
            pdf_path_str = row['PDF Path']
            # Check if the PDF file exists
            if os.path.isfile(pdf_path_str):
                # Define the destination path
                dest_path = os.path.join(destination_folder_path, os.path.basename(pdf_path_str))
                # Move the PDF file
                shutil.move(pdf_path_str, dest_path)
                # Add the PDF to the set of processed PDFs
                processed_pdfs.add(pdf_path_str)
                print(f"Moved {pdf_path_str} to {dest_path} (Category: {category})")
                # Re-check the CSV for any new updates after each move
                return True
            else:
                print(f"File does not exist and cannot be moved: {pdf_path_str}")
    return False

# Start the processing loop
while True:
    updates = process_csv()
    if not updates:
        # If no updates were processed, take a short break to avoid constant disk I/O
        time.sleep(1)
