OCR and LLM Text Processing System

This project integrates Optical Character Recognition (OCR) using EasyOCR with a Large Language Model (LLM) pipeline powered by MiniCPM-Llama3-V-2_5.
It allows users to extract text from images or PDF documents, analyze their contents, and optionally generate AI-based summaries, explanations, or contextual responses.

📂 Project Structure
project/
│
├── ocr_processing.py      # OCR extraction and document classification logic
├── app.py                 # LLM pipeline initialization and example usage
├── ocr_processing.log     # Log file (auto-generated)
└── README.md              # Project documentation

⚙️ Features
🔍 OCR Text Extraction

Extracts text from images (.png, .jpg, .jpeg, .gif) and PDF documents.

Supports multilingual OCR via EasyOCR.

Logs metadata such as file type, word count, and page count.

Optionally retrieves text positions (bounding boxes).

Detects medical documents based on domain-specific keywords.

🧠 LLM Integration

Loads the openbmb/MiniCPM-Llama3-V-2_5 model for text generation.

Supports prompt-based AI responses.

Optimized using disk offloading and device mapping for efficient memory usage.

🧰 Requirements

Before running the project, install the dependencies:

pip install easyocr pillow pdf2image numpy transformers accelerate


You may also need to install Poppler for pdf2image:

Windows: Download from Poppler for Windows

macOS: brew install poppler

Linux: sudo apt install poppler-utils

🚀 Usage
1. Run OCR Extraction

You can import and use the OCRProcessor class directly:

from ocr_processing import OCRProcessor

ocr = OCRProcessor(languages=['en'])
result = ocr.extract_text_from_input("sample.pdf")

if result['success']:
    print("Extracted Text:\n", result['text'])
else:
    print("Error:", result['error'])


To get text with bounding boxes:

positions = ocr.extract_text_with_positions("image.jpg")
print(positions)


To check if a document is medical:

is_medical = ocr.is_medical_document(result['text'])
print("Is medical document:", is_medical)

2. Run the LLM Pipeline

The app.py file initializes and tests a text-generation model.

Run it directly:

python app.py


Example expected log output:

INFO - Initializing model: openbmb/MiniCPM-Llama3-V-2_5
INFO - Tokenizer initialized successfully.
INFO - Model dispatched with device map.
INFO - Pipeline initialized successfully.
INFO - Generated text: Machine learning has revolutionized healthcare by...

🧩 Logging

All OCR processing logs are stored in:

ocr_processing.log


The log includes:

Timestamped events

File processing status

Error traces (if any)

🩺 Example Use Case

Input: Medical lab report in PDF

Output: Extracted text and LLM-generated summary explaining lab results.

🛠️ Customization

Modify the keyword list in OCRProcessor.is_medical_document() for domain adaptation.

Adjust the model name in app.py to use another LLM.

Change OCR language by updating:

OCRProcessor(languages=['en', 'fr'])
