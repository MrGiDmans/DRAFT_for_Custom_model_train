# model_train/scripts_train/train.py
import os
import torch
import json
import yaml
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from _model import MiniModel224 as DinoDragonCNN
from _model import HighModel640 as SuperDinoDragonCNN
from tqdm import tqdm
from datetime import datetime
from _utils import set_seed
from _utils.visualize_metrics import visualize_metrics
from _utils.grad_cam import generate_grad_cam

def main():
    # ======= Загрузка конфигурации =======
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg_model = config["model"]
    cfg_train = config["training"]
    cfg_data = config["data"]
    cfg_save = config["save"]
    cfg_repro = config["reproducibility"]
    cfg_resume = config["resume"]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ======= Установка сидов =======
    if cfg_repro.get("deterministic", False):
        set_seed(cfg_repro.get("seed", 42), deterministic=True)

    # ======= Загрузка и анализ классов =======
    train_path = os.path.join(cfg_data["data_path"], "train")
    train_dataset = datasets.ImageFolder(train_path)
    NUM_CLASSES = len(train_dataset.classes)

    # ======= Трансформации и загрузка =======
    transform = transforms.Compose([
        transforms.Resize((cfg_model["image_size"], cfg_model["image_size"])),
        transforms.ToTensor()
    ])

    train_dataset.transform = transform
    test_dataset = datasets.ImageFolder(os.path.join(cfg_data["data_path"], "test"), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg_train["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg_train["batch_size"], shuffle=False)

    # ======= Создание директорий =======
    model_name = cfg_train["model_name"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = cfg_save.get("base_path")

    if not base_path:
        base_path = os.path.abspath(os.path.join(SCRIPT_DIR, "./model_train/models_weights"))
    else:
        base_path = os.path.abspath(base_path)

    save_path = os.path.join(base_path, model_name, timestamp)
    os.makedirs(os.path.join(save_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "plots"), exist_ok=True)

    # ======= Логирование конфигурации =======
    with open(os.path.join(save_path, "config_used.yaml"), "w") as f:
        yaml.dump(config, f)

    # ======= Выбор модели по имени из конфигурации =======
    model_class = None
    if cfg_model["name"] == "DinoDragonCNN":
        model_class = DinoDragonCNN
    # elif cfg_model["name"] == "EnhancedDinoDragonCNN":
    #     model_class = EnhancedDinoDragonCNN
    elif cfg_model["name"] == "SuperDinoDragonCNN":
        model_class = SuperDinoDragonCNN
    else:
        raise ValueError(f"Модель {cfg_model['name']} не найдена.")

    # ======= Модель =======
    model = model_class(num_classes=NUM_CLASSES, train=True, image_size=cfg_model["image_size"]).to(DEVICE)
    
    # ======= Создание оптимизатора и loss-функции =======
    criterion = model.get_loss_function()
    optimizer = model.get_optimizer(lr=cfg_train["learning_rate"], weight_decay=cfg_train["weight_decay"], optimizer_name=cfg_train["optimizer"])

    # ======= Резюме с чекпоинта =======
    start_epoch = 0
    if cfg_resume["enabled"]:
        checkpoint = torch.load(cfg_resume["checkpoint_path"], map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1

    # ======= Обучение =======
    best_acc = 0.0
    metrics_log = {"train_loss": [], "test_accuracy": [], "confusion_matrices": []}
    log_path = os.path.join(save_path, "logs", "metrics_log.json")

    for epoch in range(start_epoch, cfg_train["epochs"]):
        model.train()
        total_loss = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg_train['epochs']}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)

            if NUM_CLASSES == 2:
                labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Losses: {avg_loss:.4f}")

        # ======= Валидация =======
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                all_preds.append(outputs)
                all_labels.append(labels)

        preds_tensor = torch.cat(all_preds)
        labels_tensor = torch.cat(all_labels)
        metrics = model.evaluate_metrics(preds_tensor, labels_tensor)

        print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        print("Error matrix:\n", metrics['confusion_matrix'])

        metrics_log["train_loss"].append(avg_loss)
        metrics_log["test_accuracy"].append(metrics['accuracy'])
        metrics_log["confusion_matrices"].append(metrics['confusion_matrix'].tolist())

        with open(log_path, "w") as f:
            json.dump(metrics_log, f, indent=4)

        # ======= Сохранение весов =======
        if cfg_save.get("save_each_epoch", True):
            torch.save(model.state_dict(), os.path.join(save_path, "checkpoints", f"model_epoch_{epoch+1}.pth"))

        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            torch.save(model.state_dict(), os.path.join(save_path, "best_weight.pth"))

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, os.path.join(save_path, "checkpoints", "last_checkpoint.pth"))

        visualize_metrics(log_path, os.path.join(save_path, "plots"))

    # === Grad-CAM после обучения ===
    grad_cam_dir = os.path.join(save_path, "Grad-CAM")
    random_images = random.sample(list(test_dataset.imgs), 5)
    sample_images = [img[0] for img in random_images]

    generate_grad_cam(
        model=model,
        image_paths=sample_images,
        output_dir=grad_cam_dir,
        target_layer_name="features.7",  # заменить на актуальное имя слоя
        class_names=train_dataset.classes,
        image_size=cfg_model["image_size"],
        device=DEVICE
    )

if __name__ == "__main__":
    main()