import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryConfusionMatrix,
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix
)

class MiniModel224(nn.Module):
    def __init__(self, num_classes=2, train=False, image_size=224):
        super(MiniModel224, self).__init__()
        self.train_mode = train
        self.num_classes = num_classes
        self.image_size = image_size
        self.binary = (num_classes == 2)

        # Сверточная часть
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        flatten_size = (image_size // 8) * (image_size // 8) * 128
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1 if self.binary else num_classes)
        )

        # Метрики
        if self.binary:
            self.accuracy = BinaryAccuracy()
            self.precision = BinaryPrecision()
            self.recall = BinaryRecall()
            self.conf_matrix = BinaryConfusionMatrix()
        else:
            self.accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')
            self.precision = MulticlassPrecision(num_classes=num_classes, average='macro')
            self.recall = MulticlassRecall(num_classes=num_classes, average='macro')
            self.conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        if self.train_mode:
            return x  # сырой логит
        else:
            if self.binary:
                return torch.sigmoid(x)  # (B, 1)
            else:
                return F.softmax(x, dim=1)  # (B, C)

    def get_loss_function(self):
        if self.binary:
            return nn.BCEWithLogitsLoss()
        else:
            return nn.CrossEntropyLoss()

    def get_optimizer(self, lr=0.001, weight_decay=0.0, optimizer_name="adam"):
        if optimizer_name.lower() == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            return torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")


    def evaluate_metrics(self, preds, labels):
        if self.binary:
            probs = torch.sigmoid(preds)
            preds_classes = (probs > 0.5).int().view(-1)
        else:
            preds_classes = torch.argmax(preds, dim=1)

        labels = labels.view(-1)

        metrics = {
            'accuracy': self.accuracy(preds_classes, labels).item(),
            'precision': self.precision(preds_classes, labels).item(),
            'recall': self.recall(preds_classes, labels).item(),
            'confusion_matrix': self.conf_matrix(preds_classes, labels).cpu().numpy()
        }
        return metrics


class HighModel640(nn.Module):
    def __init__(self, num_classes=2, image_size=640, train=False):
        super(HighModel640, self).__init__()
        self.train_mode = train
        self.num_classes = num_classes
        self.image_size = image_size
        self.binary = (num_classes == 2)

        # Более глубокие слои с пулингом и регуляризацией
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        flatten_size = (image_size // 16) * (image_size // 16) * 512
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1 if self.binary else num_classes)
        )

        # Метрики
        if self.binary:
            self.accuracy = BinaryAccuracy()
            self.precision = BinaryPrecision()
            self.recall = BinaryRecall()
            self.conf_matrix = BinaryConfusionMatrix()
        else:
            self.accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')
            self.precision = MulticlassPrecision(num_classes=num_classes, average='macro')
            self.recall = MulticlassRecall(num_classes=num_classes, average='macro')
            self.conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        if self.train_mode:
            return x  # сырой логит
        else:
            if self.binary:
                return torch.sigmoid(x)  # (B, 1)
            else:
                return F.softmax(x, dim=1)  # (B, C)

    def get_loss_function(self):
        if self.binary:
            return nn.BCEWithLogitsLoss()
        else:
            return nn.CrossEntropyLoss()

    def get_optimizer(self, lr=0.001, weight_decay=0.0, optimizer_name="adam"):
        if optimizer_name.lower() == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            return torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def evaluate_metrics(self, preds, labels):
        if self.binary:
            probs = torch.sigmoid(preds)
            preds_classes = (probs > 0.5).int().view(-1)
        else:
            preds_classes = torch.argmax(preds, dim=1)

        labels = labels.view(-1)

        metrics = {
            'accuracy': self.accuracy(preds_classes, labels).item(),
            'precision': self.precision(preds_classes, labels).item(),
            'recall': self.recall(preds_classes, labels).item(),
            'confusion_matrix': self.conf_matrix(preds_classes, labels).cpu().numpy()
        }
        return metrics

