import os
from groq import Groq
import cv2
import base64

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

def get_keyframe_descriptions(keyframes, api_key):
    """
    Generates detailed descriptions for each keyframe using Groq's API.

    Args:
        keyframes (list): A list of tuples, where each tuple contains (frame_index, frame_image).
        api_key (str): Groq API key for authentication.

    Returns:
        dict: A dictionary where keys are frame indices and values are the detailed descriptions.
    """
    client = Groq(api_key=api_key)
    descriptions = {}

    for frame_index, frame_image in keyframes:
        # Convert the frame to a base64 URL or upload it to a hosting service to get an accessible URL
        _, buffer = cv2.imencode(".jpg", frame_image)
        image_bytes = base64.b64encode(buffer).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{image_bytes}"  # Example for inline image URL

        try:
            completion = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                # "text": "Examine the image and provide a detailed summary, describing key objects, people, actions, and interactions. Include any visible text along with its context and significance. Highlight the overall scenario and message conveyed, ensuring the description integrates both visual and textual elements comprehensively and accurately.\n"
                                "text": "You are generating a detailed summary of the content present in the image by analyzing the topic of discussion in the image"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            description = completion.choices[0].message.content
            descriptions[frame_index] = description
            print(f"Generated description for frame {frame_index}")
        except Exception as e:
            print(f"Error generating description for frame {frame_index}: {e}")
            descriptions[frame_index] = "Error generating description."

    return descriptions
