import os
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

image_size = 224
augmentations_per_image = 5  # сколько аугментированных копий создать из одного оригинала

augment = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(10),
])

def gather_images(source_root, class_name):
    paths = []
    for split in ['train', 'test']:
        dir_path = os.path.join(source_root, split, class_name)
        for fname in os.listdir(dir_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                paths.append(os.path.join(dir_path, fname))
    return paths

def apply_augmentation_and_save(paths, class_name):
    images = []
    for path in tqdm(paths, desc=f"Аугментация {class_name}"):
        try:
            img = Image.open(path).convert("RGB")
            for _ in range(augmentations_per_image):
                img_aug = augment(img)
                images.append(img_aug)
        except Exception as e:
            print(f"Ошибка с {path}: {e}")
    return images

def save_images(images, target_root, class_name, split):
    save_dir = os.path.join(target_root, split, class_name)
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(images):
        img_path = os.path.join(save_dir, f"{class_name}_{i:05d}.jpg")
        img.save(img_path)

def process_class(source_root, target_root, class_name):
    all_paths = gather_images(source_root, class_name)
    all_augmented = apply_augmentation_and_save(all_paths, class_name)

    train_imgs, test_imgs = train_test_split(all_augmented, test_size=0.2, random_state=42)

    save_images(train_imgs, target_root, class_name, "train")
    save_images(test_imgs, target_root, class_name, "test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Аугментация изображений.")
    parser.add_argument("--input_path", type=str, default="data/dino-or-dragon/versions/1", help="Путь к исходному датасету.")
    parser.add_argument("--output_path", type=str, default="data/dino-or-dragon/versions/2", help="Путь для аугментированного датасета.")
    args = parser.parse_args()

    for cls in ["dino", "dragon"]:
        process_class(args.input_path, args.output_path, cls)