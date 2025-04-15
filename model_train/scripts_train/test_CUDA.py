import torch
print(torch.cuda.is_available())  # Должно быть True если используется CUDA.
print(torch.cuda.get_device_name(0))  # Должно показать какая видеокарта используется.
