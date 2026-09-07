from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
pages = sorted(ROOT.glob("page-*.png"))
thumb_width = 380
margin = 20
label_height = 32
columns = 2
rows = 4

font = ImageFont.load_default(size=18)

for sheet_index, start in enumerate(range(0, len(pages), columns * rows), start=1):
    batch = pages[start : start + columns * rows]
    prepared = []
    for path in batch:
        with Image.open(path) as source:
            image = source.convert("RGB")
            height = round(image.height * thumb_width / image.width)
            prepared.append((path, image.resize((thumb_width, height), Image.Resampling.LANCZOS)))

    cell_height = max(image.height for _, image in prepared) + label_height
    canvas = Image.new(
        "RGB",
        (
            columns * thumb_width + (columns + 1) * margin,
            rows * cell_height + (rows + 1) * margin,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for item_index, (path, image) in enumerate(prepared):
        row, column = divmod(item_index, columns)
        x = margin + column * (thumb_width + margin)
        y = margin + row * (cell_height + margin)
        draw.text((x, y), path.stem, fill="black", font=font)
        canvas.paste(image, (x, y + label_height))

    canvas.save(ROOT / f"contact-{sheet_index:02d}.jpg", quality=92)
