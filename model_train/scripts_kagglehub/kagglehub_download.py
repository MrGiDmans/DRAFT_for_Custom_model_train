import kagglehub
import os

# Укажите путь к директории, в которую хотите сохранить датасет
save_path = "./data"  # Замените на нужный путь

# Создайте директорию, если она не существует
os.makedirs(save_path, exist_ok=True)

# Загрузите датасет, указав путь сохранения
path = kagglehub.dataset_download("agrigorev/dino-or-dragon", destdir=save_path)

print("Путь к файлам датасета:", path)