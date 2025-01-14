import os
from dotenv import load_dotenv
from modules.data_processing import detect_keyframes
from modules.visualization import save_keyframes
from modules.models import load_whisper_model,transcribe_audio
from modules.summarization import summarize_with_groq,get_keyframe_descriptions
VIDEO_PATH = "sample/example2.mp4"
OUTPUT_FOLDER = "outputs/keyframes"
load_dotenv()
api = os.getenv("GROQ_API_KEY")

keyframes = detect_keyframes(VIDEO_PATH)
keyframes_description = get_keyframe_descriptions(keyframes,api_key=api)
for i in keyframes_description:
    print(keyframes_description[i])
# save_keyframes(keyframes, OUTPUT_FOLDER)
# display_keyframes(keyframes)

# Load the models
# whisper_model = load_whisper_model("base")

# Transcribe audio
# audio_transcript = transcribe_audio(VIDEO_PATH, whisper_model)

# print("### Audio Transcript ###")
# print(audio_transcript)

# audio_summary = summarize_with_groq(audio_transcript,api_key=api)
# print("---------------------AUDIO BASED SUMMARY-------------------------")
# print(audio_summary)
# Verify if models are loaded
# print("Models loaded successfully!")
# print("check for git")