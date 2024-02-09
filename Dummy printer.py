import time
import os
import fitz  # PyMuPDF
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class PrintFolderHandler(FileSystemEventHandler):
    def on_created(self, event):
        # Only process the file if it's a PDF
        if event.is_directory or not event.src_path.lower().endswith('.pdf'):
            return
        
        print(f"Detected new PDF: {event.src_path}. Starting simulated printing process...")
        self.print_with_loading_bar(event.src_path, 10)  # 10-second loading bar
        self.log_print_success(event.src_path)

    def print_with_loading_bar(self, file_path, duration):
        # Print a loading bar that fills up over the specified duration (in seconds)
        increments = 50
        for i in range(increments + 1):
            time.sleep(duration / increments)
            progress = int((i / increments) * 100)
            bar = '#' * int((i / increments) * 50)
            print(f"\rPrinting {os.path.basename(file_path)}: [{bar:<50}] {progress}%", end='')
        print()  # Newline after loading bar completes

    def log_print_success(self, pdf_path):
        # Log the successful print action
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open("print_log.txt", "a") as log_file:
            log_file.write(f"[{timestamp}] Successfully printed {pdf_path}\n")
        print(f"Successfully printed {os.path.basename(pdf_path)}.")

def start_print_folder(path):
    event_handler = PrintFolderHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    print(f"Monitoring folder: {path} for new PDF files to simulate printing...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    # Set the directory you want to monitor
    print_folder_path = r"C:\Users\Sri Vishnu V S\Desktop\Printer"
    start_print_folder(print_folder_path)
