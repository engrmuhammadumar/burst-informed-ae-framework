from pathlib import Path
from PIL import Image

if __name__ == "__main__":
    for split in ["train","val","test"]:
        p = Path("outputs")/f"confmat_{split}.png"
        if p.exists():
            Image.open(p).show()
