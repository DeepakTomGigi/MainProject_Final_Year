import os
from groq import Groq

# def combine_modalities(frame_descriptions, audio_transcript):
#     """
#     Combines frame descriptions and audio transcripts into a single input for summarization.

#     Args:
#         frame_descriptions (list): List of descriptions for keyframes.
#         audio_transcript (str): Transcribed text from the video's audio.

#     Returns:
#         str: Combined text input.
#     """
#     combined_input = "### Video Summary\n"
#     combined_input += "\n".join(frame_descriptions)
#     combined_input += f"\n\n### Audio Transcript\n{audio_transcript}"
#     return combined_input


def summarize_with_groq(audio_transcript, api_key):
    """
    Summarizes the audio transcript using Groq's API.

    Args:
        audio_transcript (str): Transcribed text from the video's audio.
        api_key (str): Groq API key for authentication.

    Returns:
        str: Generated summary text from Groq.
    """
    client = Groq(api_key=api_key)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are generating a transcript summary. Provide a detailed and coherent summary of the given transcript."},
                {"role": "user", "content": audio_transcript},
            ],
            temperature=0,
            model="llama3-8b-8192",
        )
        summary = chat_completion.choices[0].message.content
        return summary
    except Exception as e:
        print(f"Error while generating summary with Groq: {e}")
        return "Summary generation failed."
