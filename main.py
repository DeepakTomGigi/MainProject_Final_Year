from modules.data_processing import detect_keyframes
from modules.visualization import save_keyframes
from modules.models import load_whisper_model,transcribe_audio
from modules.summarization import summarize_with_groq
VIDEO_PATH = "sample/example2.mp4"
OUTPUT_FOLDER = "outputs/keyframes"

keyframes = detect_keyframes(VIDEO_PATH)
save_keyframes(keyframes, OUTPUT_FOLDER)
# display_keyframes(keyframes)

# Load the models
whisper_model = load_whisper_model("base")

# Transcribe audio
audio_transcript = transcribe_audio(VIDEO_PATH, whisper_model)

print("### Audio Transcript ###")
print(audio_transcript)
# blip_model, blip_processor = load_blip_model()
audio_summary = summarize_with_groq(audio_transcript,"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")
print("---------------------AUDIO BASED SUMMARY-------------------------")
print(audio_summary)
# Verify if models are loaded
print("Models loaded successfully!")
