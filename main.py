import os
from dotenv import load_dotenv
from modules.data_processing import detect_keyframes
from modules.visualization import save_keyframes
from modules.models import load_whisper_model,transcribe_audio
from modules.summarization import summarize_with_groq,get_keyframe_descriptions
from modules.presentation import generate_presentation

VIDEO_PATH = "sample/whatisml.mp4"
OUTPUT_FOLDER = "outputs/keyframes"
load_dotenv()
api = os.getenv("GROQ_API_KEY")

keyframes = detect_keyframes(VIDEO_PATH)
keyframes_description = get_keyframe_descriptions(keyframes[:5],api_key=api)
final_prompt = "The descriptions given below are the description of each keyframes of the video.some descriptions might be similar so learn only necessary details without repitition.\n"

i = 1
for descriptions in keyframes_description:
    final_prompt += "Keyframe "+str(i)+":" + keyframes_description[descriptions] + "\n"
    i+=1
    
save_keyframes(keyframes, OUTPUT_FOLDER)
# display_keyframes(keyframes)

# Load the models
whisper_model = load_whisper_model("base")

# Transcribe audio
audio_transcript = transcribe_audio(VIDEO_PATH, whisper_model)

final_prompt += "The text below will be the audio transcript of the video\n"
final_prompt += audio_transcript
# print(final_prompt)

final_summary = summarize_with_groq(final_prompt,api_key=api)
print("\n\n---------------------SUMMARY-------------------------\n\n")
print(final_summary)

generate_presentation(final_summary)

# print("\n\n---------------------SUMMARY-------------------------\n\n")
# print(final_summary)