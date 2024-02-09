# Hello, we are the Code Crafters

## Description

In the fast-paced digital world, efficient management of PDF documents is a cornerstone for businesses and organizations in various industries. Our project introduces a systematic workflow to manage PDF documents effectively, integrating data preprocessing, categorization, priority scheduling, and printing. These strategies are designed to streamline document management, prioritize critical information, and boost productivity. Dive into our workflow and explore the key components that make it work.

## Getting Started

### Dependencies

Ensure you have the following Python modules installed:

- `torch` - A machine learning library for high performance with tensors and dynamic neural networks.
- `pandas` - Essential for data manipulation and analysis.
- `transformers` - Offers numerous pre-trained models for Natural Language Understanding (NLU) and Generation (NLG).
- `sklearn` (scikit-learn) - Provides tools for data mining and analysis.
- `os` - Interfaces with the operating system and is part of the Python Standard Library.
- `joblib` - Useful for lightweight pipelining and working with large datasets.
- `fitz` (PyMuPDF) - Enables creation and manipulation of PDF files.
- `shutil` - Facilitates high-level file operations, such as copying and archiving.

Install the necessary modules with the following command (note that `os` and `shutil` are included with Python and do not need to be installed):

``` pip install torch pandas transformers scikit-learn joblib PyMuPDF ```

### Installing
Clone the repository and install the dependencies as mentioned above. You may need to adjust file paths in the configuration files to match your local setup.

### Executing the Program
Follow these steps to run the project:

Step 1: Training the Model
Train the machine learning model with your dataset by executing:

``` python train.py ```

Step 2: Testing the Model
Evaluate the model's performance using the testing script:

``` python test+.py ```

Step 3: Prioritizing Files
Finally, prioritize the files for printing based on the model's assigned priority:

``` python prioritize+.py ```

Step 4: Implementing it on a dummy printer
After prioritizing the files, the next step is to simulate the printing process on a dummy printer. This step is crucial for testing the end-to-end workflow without using actual printer resources.

``` python dummy printer.py ```

### Contributors
A heartfelt thanks to all the contributors who have made this project possible:

- D Ankith - https://github.com/ankithdadda
- Poornachandra A N - https://github.com/Heisenberg208
- Sri Vishnu VS - https://github.com/srivishnuvs
- Yashwanth M - https://github.com/yashwanthm998
