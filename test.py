# import torch
# from dotenv import load_dotenv

# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
# print("cuDNN Enabled:", torch.backends.cudnn.enabled)

import os
from flask import Flask, request, jsonify, render_template
from PIL import Image
import pytesseract
import re
from transformers import BartForConditionalGeneration, BartTokenizer
import assemblyai as aai
from extractor import extract_information_from_text

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(_name_)

aai.settings.api_key = "ae3c31ffff904933a071a99e6477b7f3"  

tokenizer = BartTokenizer.from_pretrained("./models/bart-fine-tuned-mts")
model = BartForConditionalGeneration.from_pretrained("./models/bart-fine-tuned-mts")

def clean_ocr_text(ocr_text):
    lines = ocr_text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line: 
            continue
        if re.match(r"^\d{5,}$", line): 
            continue
        if re.search(r"@|\.com|www", line, re.IGNORECASE):  
            continue
        if re.match(r"^Page\s\d+", line, re.IGNORECASE):  
            continue
        if len(line.split()) <= 2 and re.search(r"\d{2,4}", line):  
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    extracted_text = pytesseract.image_to_string(Image.open(image))
    cleaned_text = clean_ocr_text(extracted_text)

    extracted_info = extract_information_from_text(cleaned_text)

    print("\n--- Extracted Information ---")
    for key, value in extracted_info.items():
        print(f"{key}: {value}")
    print("-----------------------------\n")
    return jsonify({"extracted_text": cleaned_text})

@app.route('/summarize', methods=['POST'])
def summarize(): 
    ocr_text = request.json.get("ocr_text", "")
    additional_text = request.json.get("additional_text", "")
    audio_text = request.json.get("transcription", "")

    if not ocr_text and not additional_text and not audio_text:
        return jsonify({"error": "No input provided"}), 400

    # Extract structured information from each input
    ocr_info = extract_information_from_text(ocr_text) if ocr_text else {}
    additional_info = extract_information_from_text(additional_text) if additional_text else {}
    audio_info = extract_information_from_text(audio_text) if audio_text else {}

    # Combine extracted info into a structured clinical note
    combined_info = {
        "Name": ocr_info.get("Name") or additional_info.get("Name") or audio_info.get("Name"),
        "Age": ocr_info.get("Age") or additional_info.get("Age") or audio_info.get("Age"),
        "Gender": ocr_info.get("Gender") or additional_info.get("Gender") or audio_info.get("Gender"),
        "Disease": ocr_info.get("Disease") or additional_info.get("Disease") or audio_info.get("Disease"),
        "Medications": list(set(
            (ocr_info.get("Medications") or []) +
            (additional_info.get("Medications") or []) +
            (audio_info.get("Medications") or [])
        )),
        "Discharge Date": ocr_info.get("Discharge Date") or additional_info.get("Discharge Date") or audio_info.get("Discharge Date")
    }

    # Create a structured prompt for BART
    combined_text = (
        "Generate a discharge summary based on the following clinical information:\n"
        f"Patient Name: {combined_info['Name'] or 'Not specified'}\n"
        f"Age: {combined_info['Age'] or 'Not specified'}\n"
        f"Gender: {combined_info['Gender'] or 'Not specified'}\n"
        f"Disease/Condition: {combined_info['Disease'] or 'Not specified'}\n"
        f"Medications: {', '.join(combined_info['Medications']) or 'None specified'}\n"
        f"Discharge Date: {combined_info['Discharge Date']}\n"
        "Additional Notes from Image Report:\n" + (ocr_text if ocr_text else "None provided") + "\n"
        "Additional Notes from Text Input:\n" + (additional_text if additional_text else "None provided") + "\n"
        "Additional Notes from Doctor-Patient Conversation:\n" + (audio_text if audio_text else "None provided") + "\n"
        "Provide a concise discharge summary incorporating all relevant details."
    ).strip()

    # Tokenize and generate summary with adjusted parameters
    inputs = tokenizer(combined_text, return_tensors="pt", max_length=4096, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=900,  # Adjust based on desired summary length
        min_length=100,  # Ensure a minimum length for completeness
        num_beams=4,
        early_stopping=True,
        length_penalty=1.0  # Neutral length penalty to avoid over-shortening
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = re.sub(r'\s+', ' ', summary).strip()

    return jsonify({"summary": summary})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if audio_file:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        audio_path = os.path.join('uploads', audio_file.filename)
        audio_file.save(audio_path)

        try:
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_path)

            if transcript.status == aai.TranscriptStatus.error:
                return jsonify({"error": transcript.error}), 500
            else:
                return jsonify({"transcription": transcript.text})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            try:
                os.remove(audio_path)
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}")

@app.route('/')
def index():
    return render_template('index.html')

if _name_ == '_main_':
    app.run(debug=True)