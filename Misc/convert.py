from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Load your TTF font at 8px size
font = ImageFont.truetype("Code8x8.ttf", 4)

for char in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
    img = Image.new("L", (4, 4), 0)  # Black background
    draw = ImageDraw.Draw(img)
    draw.text((0, -1), char, 255, font=font)  # White text

    # Convert to binary (0/1)
    arr = np.array(img)
    binary = (arr > 128).astype(int)
    print(f"// {char}")
    print(binary.flatten().tolist())
    img.save(f"{char}.png")
