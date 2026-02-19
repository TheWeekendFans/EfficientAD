import torch

print("python ok")
print("torch=", torch.__version__)
print("cuda=", torch.cuda.is_available())
print("gpus=", torch.cuda.device_count())

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print("gpu0=", torch.cuda.get_device_name(0))
else:
    print("gpu0=", "N/A")
