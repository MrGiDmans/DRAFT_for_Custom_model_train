# model_train/_utils/visualize_metrics.py

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def visualize_metrics(log_path: str, output_dir: str):
    with open(log_path, "r") as f:
        log = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(log["train_loss"], label="Train Loss")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title("График потерь")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "train_loss.png"))
    plt.close()

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(log["test_accuracy"], label="Test Accuracy")
    plt.xlabel("Эпоха")
    plt.ylabel("Accuracy")
    plt.title("График точности")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "test_accuracy.png"))
    plt.close()

    # Confusion matrix (последняя)
    cm = np.array(log["confusion_matrices"][-1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (последняя эпоха)")
    plt.xlabel("Предсказано")
    plt.ylabel("Истинное")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    print(f"[✓] Графики сохранены в {output_dir}")

# Возможность запускать отдельно, если нужно
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True, help="Путь до metrics_log.json")
    parser.add_argument("--output_dir", type=str, default="visualization", help="Куда сохранить графики")
    args = parser.parse_args()
    visualize_metrics(args.log_path, args.output_dir)
