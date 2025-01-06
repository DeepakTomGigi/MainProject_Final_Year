# import subprocess

# try:
#     result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
#     print(result.stdout)
# except FileNotFoundError:
#     print("ffmpeg is not accessible.")

# from modules.models import load_whisper_model
# model = load_whisper_model("base")
# print("sucess")
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print("cuDNN Enabled:", torch.backends.cudnn.enabled)