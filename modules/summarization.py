import os
from groq import Groq
import cv2
import base64




def  summarize_with_groq(final_input, api_key):
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
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": "You will be provided the keyframe descriptions and audio transcript of a video. Generate meaningful summary in the below format\n\"\"\"\n[Title]\nMy Presentation\nA Subtitle\n\n[Content]\nIntroduction\n- Point 1 is a brief statement\n- Point 2 has some more details to explain briefly\nMain Topic\n- Detail A is a short point\n- Detail B is longer and requires more explanation to fully understand the concept\n- Detail C continues with additional information that might overflow a single slide\nAnother Section\n- Point X\n- Point Y with a longer explanation that could span multiple lines in a presentation slide\n\n[Conclusion]\nSummary of key points\n- Final thought\n\"\"\"\nimportant points to note:\nexplain the main topics as much as possible there should be content for minimum 10 pages and give output in standard format do not  make anything bold.\nthere should always be a [Title],[Content],[Conclusion] also the subheadings should not use any square brackets.\nThe output should always begin with [Title] strictly."
            },
            {
                "role": "user",
                "content": final_input
            },
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
        summary = ""
        for chunk in chat_completion:
            # print(chunk.choices[0].delta.content or "", end="")
            summary += chunk.choices[0].delta.content or ""
        print("\n")
        print(len(summary))
        print(summary)
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
                                "text": "Summarize the key frame by identifying the main educational concepts present in the image, text,equations or diagrams in the image. Keep the summary brief but ensure it captures all essential details excluding unimportant information without excessive description."
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