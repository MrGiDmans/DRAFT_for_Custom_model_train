import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import Image

def generate_grad_cam(model, image_paths, output_dir, target_layer_name, class_names, image_size, device):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    def preprocess_image(img_path):
        image = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        return transform(image).unsqueeze(0).to(device), image

    def apply_colormap_on_image(org_img, activation_map, colormap_name='jet'):
        heatmap = cv2.applyColorMap(np.uint8(255 * activation_map), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(org_img) / 255
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)

    # Получение последнего слоя по имени
    target_layer = dict([*model.named_modules()])[target_layer_name]

    gradients = None
    activations = None

    def save_gradients_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    def save_activations_hook(module, input, output):
        nonlocal activations
        activations = output

    target_layer.register_forward_hook(save_activations_hook)
    target_layer.register_backward_hook(save_gradients_hook)

    for img_path in image_paths:
        input_tensor, original_image = preprocess_image(img_path)

        input_tensor.requires_grad = True
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        model.zero_grad()
        class_score = output[0, pred_class]
        class_score.backward()

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap = cv2.resize(heatmap, (original_image.width, original_image.height))
        heatmap = heatmap / heatmap.max()

        cam_result = apply_colormap_on_image(np.array(original_image), heatmap)
        result_path = os.path.join(output_dir, os.path.basename(img_path).split('.')[0] + f"_cam_{class_names[pred_class]}.jpg")
        cv2.imwrite(result_path, cv2.cvtColor(cam_result, cv2.COLOR_RGB2BGR))

    print(f"[✓] Grad-CAM изображения сохранены в {output_dir}")
