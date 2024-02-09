import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import fitz  # PyMuPDF for PDF text extraction
import joblib  # For loading the LabelEncoder
import pandas as pd

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    return text

# Function to update the CSV file with new results
def update_csv_with_new_data(existing_df, new_results_df, csv_save_path):
    try:
        # Check if any of the new PDF paths already exist in the existing DataFrame
        existing_pdf_paths = existing_df['PDF Path'].tolist()
        new_pdf_paths = new_results_df['PDF Path'].tolist()
        
        # Filter out any new results that are already in the existing DataFrame
        new_results_df = new_results_df[~new_results_df['PDF Path'].isin(existing_pdf_paths)]
        
        # Concatenate the new results with the existing DataFrame
        updated_df = pd.concat([existing_df, new_results_df], ignore_index=True)
        
        # Save the updated DataFrame to the CSV file
        updated_df.to_csv(csv_save_path, index=False)
        print("Updated CSV file with new data.")
    except PermissionError as e:
        print(f"PermissionError: {e}. Please ensure the file is not open in another program and that you have write permissions.")

# Load the saved model and tokenizer
save_directory = r"C:\Users\Ankith Jain\OneDrive\Desktop\adobe hack\cleaned"
tokenizer = AutoTokenizer.from_pretrained(save_directory)
loaded_model = AutoModelForSequenceClassification.from_pretrained(save_directory)

# Move the loaded model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)

# Load the LabelEncoder
label_encoder_path = r"C:\Users\Ankith Jain\OneDrive\Desktop\adobe hack\cleaned\label_encoder.joblib"
label_encoder = joblib.load(label_encoder_path)

# Load your PDF test data
pdf_directory = r"C:\Users\Ankith Jain\Downloads\Test Data"
pdf_files = [file for file in os.listdir(pdf_directory) if file.endswith(".pdf")]

# Specify the path to the existing CSV file (make sure it ends with .csv)
csv_save_path = r"C:\Users\Ankith Jain\Desktop\Categorisation\results.csv"

# Check if the CSV file already exists
if os.path.exists(csv_save_path):
    try:
        # Load the existing DataFrame from the CSV file
        existing_df = pd.read_csv(csv_save_path)
    except PermissionError as e:
        print(f"PermissionError: {e}. Please ensure the file is not open in another program and that you have write permissions.")
        # Exit the script or handle the error as needed
        exit()
else:
    # Create an empty DataFrame if the CSV file doesn't exist
    columns = ["PDF Path"] + [f"Category_{i+1}" for i in range(10)]
    existing_df = pd.DataFrame(columns=columns)
    # Save the empty DataFrame as a new CSV file
    existing_df.to_csv(csv_save_path, index=False)

# Create an empty DataFrame to store new results
columns = ["PDF Path"] + [f"Category_{i+1}" for i in range(10)]
new_results_df = pd.DataFrame(columns=columns)

# Custom dataset class for text classification
class TestTextClassificationDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

# Loop through PDFs and make predictions
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    test_text = extract_text_from_pdf(pdf_path)

    # Create a test dataset
    test_texts = [test_text]
    test_dataset = TestTextClassificationDataset(test_texts, tokenizer)

    # Example inference loop
    loaded_model.eval()  # Set the model to evaluation mode
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = loaded_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Assuming a classification task with softmax activation
            probabilities = torch.softmax(logits, dim=1)
            predicted_label_num = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
            predictions.append(predicted_label_num)

            # Convert numerical prediction back to the original class name
            predicted_label_str = label_encoder.inverse_transform([predicted_label_num])[0]
            print(f"Predicted label for the input text: {predicted_label_str}")

    # Store results in the new DataFrame
    result_row = {"PDF Path": pdf_path}
    for i in range(10):
        result_row[f"Category_{i+1}"] = 1 if i == predicted_label_num else 0

    new_results_df = pd.concat([new_results_df, pd.DataFrame([result_row])], ignore_index=True)

# Update the CSV file with new data
try:
    update_csv_with_new_data(existing_df, new_results_df, csv_save_path)
except PermissionError as e:
    print(f"PermissionError: {e}. Please ensure the file is not open in another program and that you have write permissions.")