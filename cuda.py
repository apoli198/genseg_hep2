import torch
print(torch.cuda.is_available())       # True if GPU is usable
print(torch.version.cuda)              # CUDA version PyTorch was built against
print(torch.cuda.get_device_name(0))   # Name of your first GPU
print(torch.__version__)               # PyTorch version
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
    print("GPU supports mixed precision training")