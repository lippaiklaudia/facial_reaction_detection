import argparse
from src.data_preprocessing import load_data
from src.train_model import train_model

def main(task):
    data_dir = "data/fer2013"

    if task == "train":
        print("Adatok betöltése...")
        train_data, test_data = load_data(data_dir)

        print("Modell betanítása...")
        model_path = "models/emotion_detection_model.h5"
        train_model(train_data, test_data, model_path)

    elif task == "test":
        print("A teszt funkció még nincs implementálva.")
    else:
        print(f"Ismeretlen task: {task}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FER2013 érzelemfelismerő rendszer.")
    parser.add_argument(
        "task",
        type=str,
        choices=["train", "test"],
        help="Feladat: 'train' a modell betanításához, 'test' a teszteléshez."
    )
    args = parser.parse_args()

    main(args.task)
