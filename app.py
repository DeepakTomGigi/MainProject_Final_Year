from flask_cors import CORS
import os
from modules.data_processing import detect_keyframes
from modules.models import load_whisper_model, transcribe_audio
from modules.presentation import generate_presentation
from modules.summarization import get_keyframe_descriptions, summarize_with_groq
from dotenv import load_dotenv
from flask import Flask, jsonify, send_file, request
import uuid
import time

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "*"}})

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# VIDEO_PATH = "sample/example2.mp4"

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the backend API. Use /upload to upload videos."})

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/upload", methods=["POST"])
def upload_video():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        video_file = request.files["video"]
        video_path = os.path.join("sample", video_file.filename)
        os.makedirs("sample", exist_ok=True)  # Save videos in "sample" directory
        video_file.save(video_path)
        
        # Brief delay to ensure file is fully written (optional)
        time.sleep(1)
        
        if not os.path.exists(video_path):
            return jsonify({"error": "Video file not saved properly"}), 500
        
        # Process the video
        print("Video path------>", video_path)
        keyframes = detect_keyframes(video_path)
        keyframes_description = get_keyframe_descriptions(keyframes, api_key=api_key)
        final_prompt = "Keyframe Descriptions:\n"
        for i, desc in enumerate(keyframes_description.values(), 1):
            final_prompt += f"Keyframe {i}: {desc}\n"
        
        whisper_model = load_whisper_model("base")
        audio_transcript = transcribe_audio(video_path, whisper_model)
        final_prompt += f"\nAudio Transcript:\n{audio_transcript}"
        
        final_summary = summarize_with_groq(final_prompt, api_key=api_key)
        
        # Generate and save the PowerPoint presentation
        # ppt_filename = f"{uuid.uuid4()}.pptx"

        # ppt_dir = os.path.join("static", "presentations")
        # os.makedirs(ppt_dir, exist_ok=True)
        # ppt_path = os.path.join(ppt_dir, ppt_filename)
        ppt_filename= generate_presentation(final_summary)
        
        ppt_url = f"/{ppt_filename}"
        
        return jsonify({
            "summary": final_summary,
            "ppt_url": ppt_url
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/presentations/<filename>')
def download_ppt(filename):
    ppt_dir = "static/presentations"
    ppt_path = os.path.join(ppt_dir, filename)
    if not os.path.exists(ppt_path):  # Check if the file exists
        return jsonify({"error": "File not found"}), 404

    return send_file(ppt_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)