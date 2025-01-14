import torch
from dotenv import load_dotenv

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print("cuDNN Enabled:", torch.backends.cudnn.enabled)
