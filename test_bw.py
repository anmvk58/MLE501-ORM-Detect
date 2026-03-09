from pathlib import Path
from PIL import Image

folder = Path(r"D:\Coding\MSE35HN\MLE501\data_cleaned\D")

for img_path in folder.glob("*"):
    if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        with Image.open(img_path) as img:
            width, height = img.size

        if width > 250 or width < 200:
            img_path.unlink()
            print(f"Deleted {img_path.name}")