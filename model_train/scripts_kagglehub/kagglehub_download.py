import kagglehub
import os

def main():
    # Укажите путь к директории, в которую хотите сохранить датасет
    save_path = "./data1"  # Замените на нужный путь

    # Создайте директорию, если она не существует
    os.makedirs(save_path, exist_ok=True)

    # Загрузите датасет, указав путь сохранения
    path = kagglehub.dataset_download("agrigorev/dino-or-dragon", path=save_path)

    print("Путь к файлам датасета:", path)

if __name__ == "__main__":
    main()
