import whisper
import moviepy
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


def load_whisper_model(model_size="base"):
    """
    Loads the Whisper model for audio transcription.

    Args:
        model_size (str): The size of the Whisper model to load (e.g., "tiny", "base", "small", "medium", "large").

    Returns:
        model: The loaded Whisper model.
    """
    print(f"Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size,device="cuda")
    return model

from moviepy import VideoFileClip

def extract_audio_from_video(video_path, audio_output_path="temp_audio.wav"):
    """
    Extracts audio from a video file and saves it as a WAV file.

    Args:
        video_path (str): Path to the input video file.
        audio_output_path (str): Path to save the extracted audio file.

    Returns:
        str: Path to the saved audio file.
    """
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_output_path)
    print(f"Audio extracted and saved to: {audio_output_path}")
    return audio_output_path


def transcribe_audio(video_path, model):
    """
    Transcribes audio using a Whisper model.

    Args:
        audio_path (str): Path to the audio file to transcribe.
        model: Preloaded Whisper model.

    Returns:
        str: Transcribed text.
    """
    video_clip = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video_clip.audio.write_audiofile(audio_path)

    result = model.transcribe(audio_path)
    return result["text"]



def load_blip_model():
    """
    Loads the BLIP model and processor for image captioning.

    Returns:
        tuple: The BLIP model and processor.
    """
    print("Loading BLIP model for image captioning...")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    return blip_model, blip_processor
