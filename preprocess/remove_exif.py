from PIL import Image, ImageOps
import glob


files = sorted(glob.glob("./B/*"))

for file_name in files:
    img = Image.open(file_name)

    if (img._getexif() is not None):
        image = ImageOps.exif_transpose(img)
        image.save(file_name)
