import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


GOST_FORMS = {
    "FORM_3": {"name": "Форма 3", "width_mm": 185, "height_mm": 55},
}

PAPER_SIZES = {
    "A3": (297, 420),
    "A4": (210, 297),
}


def mm_to_pixels(mm: float, dpi: int) -> int:
    return int(mm * dpi / 25.4)


def get_font_path() -> str:
    fonts = [
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in fonts:
        if Path(fp).exists():
            return fp
    return ""


def draw_russian_text(img_array: np.ndarray, text: str, pos: Tuple[int, int], font_size: int) -> np.ndarray:
    img_pil = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(get_font_path(), font_size) if get_font_path() else ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=(0, 0, 0))
    return np.array(img_pil)


def create_stamp_grid(form_name: str, dpi: int) -> np.ndarray:
    form = GOST_FORMS[form_name]
    w = mm_to_pixels(form["width_mm"], dpi)
    h = mm_to_pixels(form["height_mm"], dpi)
    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    rows, cols = 5, 9
    cell_w = w // cols
    cell_h = h // rows
    thickness = max(1, dpi // 100)

    for i in range(rows + 1):
        cv2.line(img, (0, i * cell_h), (w, i * cell_h), (0, 0, 0), thickness)
    for j in range(cols + 1):
        cv2.line(img, (j * cell_w, 0), (j * cell_w, h), (0, 0, 0), thickness)

    font_size = max(28, dpi // 8)
    texts = ["ИЗМ", "Лист", "№ докум.", "Подпись", "Дата"]
    for i, text in enumerate(texts[:rows]):
        img = draw_russian_text(img, text, (cell_w // 4, i * cell_h + cell_h // 2 - font_size // 2), font_size)

    for j in range(min(cols, 3)):
        img = draw_russian_text(img, str(j + 1), (j * cell_w + cell_w // 2 - 10, cell_h // 2 - font_size // 2), font_size)

    for i in range(rows):
        for j in range(cols):
            if random.random() < 0.3:
                txt = random.choice(["", "А", "Б", "В", "Г", "1", "2", "3"])
                if txt:
                    img = draw_russian_text(img, txt, (j * cell_w + cell_w // 4, i * cell_h + cell_h // 2 - font_size // 2), font_size)

    return img


def add_drawing_content(img: np.ndarray, dpi: int) -> np.ndarray:
    h, w = img.shape[:2]
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    color = (random.randint(100, 180),) * 3

    for _ in range(50):
        el = random.choice(["line", "circle", "rect"])
        if el == "line":
            x1, y1 = random.randint(50, w-50), random.randint(50, h-50)
            x2, y2 = random.randint(50, w-50), random.randint(50, h-50)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=max(1, dpi//300))
        elif el == "circle":
            r = random.randint(20, min(w, h)//8)
            cx, cy = random.randint(100, w-100), random.randint(100, h-100)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=color, width=2)
        else:
            x, y = random.randint(50, w-100), random.randint(50, h-100)
            draw.rectangle([x, y, x+random.randint(50, 200), y+random.randint(50, 200)], outline=color, width=2)

    return np.array(img_pil)


def add_artifacts(img: np.ndarray, level: str = "medium") -> np.ndarray:
    s, b, r = {"low": (3, 0.2, 1), "medium": (5, 0.3, 2), "high": (10, 0.5, 3)}.get(level, (5, 0.3, 2))

    noise = np.random.normal(0, s, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    if random.random() < b:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    angle = random.uniform(-r, r)
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    return img


def generate_synthetic_image(
    form_name: str = None,
    dpi: int = 200,
    paper_size: str = None,
    corner: str = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    form_name = form_name or random.choice(list(GOST_FORMS.keys()))
    paper_size = paper_size or random.choice(list(PAPER_SIZES.keys()))
    corner = corner or random.choice(["bottom_right", "bottom_left", "top_right", "top_left"])

    form = GOST_FORMS[form_name]
    w_mm, h_mm = PAPER_SIZES[paper_size]

    canvas_w = mm_to_pixels(w_mm, dpi)
    canvas_h = mm_to_pixels(h_mm, dpi)

    if random.random() < 0.5:
        canvas_w, canvas_h = canvas_h, canvas_w

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    canvas = add_drawing_content(canvas, dpi)

    stamp = create_stamp_grid(form_name, dpi)
    stamp_h, stamp_w = stamp.shape[:2]

    margin = max(20, dpi // 4)
    corners = {
        "bottom_right": (max(0, canvas_w - stamp_w - margin), max(0, canvas_h - stamp_h - margin)),
        "bottom_left": (margin, max(0, canvas_h - stamp_h - margin)),
        "top_right": (max(0, canvas_w - stamp_w - margin), margin),
        "top_left": (margin, margin),
    }
    x, y = corners[corner]

    stamp_h_f = min(stamp_h, canvas_h - y)
    stamp_w_f = min(stamp_w, canvas_w - x)
    canvas[y:y+stamp_h_f, x:x+stamp_w_f] = stamp[:stamp_h_f, :stamp_w_f]

    canvas = add_artifacts(canvas, random.choice(["low", "medium", "high"]))

    label = np.array([
        0,
        (x + stamp_w_f / 2) / canvas_w,
        (y + stamp_h_f / 2) / canvas_h,
        stamp_w_f / canvas_w,
        stamp_h_f / canvas_h
    ])

    metadata = {
        "form_name": form_name,
        "form_width_mm": form["width_mm"],
        "form_height_mm": form["height_mm"],
        "dpi": dpi,
        "paper_size": paper_size,
        "corner": corner,
        "stamp_bbox": [x, y, stamp_w_f, stamp_h_f],
    }

    return canvas, label, metadata


def crop_stamp_from_image(image: np.ndarray, labels: np.ndarray, margin: float = 0.05) -> np.ndarray:
    h, w = image.shape[:2]
    for label in labels:
        _, cx, cy, bw, bh = label
        x = int((cx - bw / 2) * w)
        y = int((cy - bh / 2) * h)
        cw = int(bw * w)
        ch = int(bh * h)
        
        mx = int(cw * margin)
        my = int(ch * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + cw + mx)
        y2 = min(h, y + ch + my)
        return image[y1:y2, x1:x2]
    return np.array([])


def clean_background(image: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Fill stamp bbox with local noise to erase original stamp."""
    h, w = image.shape[:2]
    _, cx, cy, bw, bh = label
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return image.copy()
    roi = image[y1:y2, x1:x2]
    mean = roi.mean(axis=(0, 1))
    std = roi.std(axis=(0, 1))
    noise = np.clip(np.random.normal(mean, std, roi.shape), 0, 255).astype(np.uint8)
    cleaned = image.copy()
    cleaned[y1:y2, x1:x2] = noise
    return cleaned


def resize_stamp_for_bg(
    stamp: np.ndarray,
    bg_w: int,
    bg_h: int,
    max_rel_width: float = 0.6,
    max_rel_height: float = 0.25,
    max_upscale: float = 3.0,
) -> np.ndarray:
    """Scale stamp to fit background proportionally.

    Pasted stamp occupies up to `max_rel_width` of background width
    and up to `max_rel_height` of background height. Never upscales
    more than `max_upscale`× to avoid blurring.
    """
    s_h, s_w = stamp.shape[:2]
    target_w = int(bg_w * max_rel_width)
    target_h = int(target_w * (s_h / s_w))
    max_h = int(bg_h * max_rel_height)
    if target_h > max_h:
        scale = max_h / target_h
        target_w = int(target_w * scale)
        target_h = max_h
    scale = min(target_w / s_w, target_h / s_h, max_upscale)
    scale = max(scale, 0.1)
    new_w = max(1, int(s_w * scale))
    new_h = max(1, int(s_h * scale))
    if new_w == s_w and new_h == s_h:
        return stamp
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(stamp, (new_w, new_h), interpolation=interp)


def place_by_orientation(
    bg_w: int, bg_h: int, stamp_w: int, stamp_h: int, margin: int = 10
) -> tuple:
    """Place stamp based on sheet orientation."""
    if bg_w >= bg_h:  # landscape → bottom-right
        x = max(0, bg_w - stamp_w - margin)
    else:  # portrait → bottom-center
        x = max(0, (bg_w - stamp_w) // 2)
    y = max(0, bg_h - stamp_h - margin)
    return x, y


def generate_synthetic_from_real(
    background_img: np.ndarray,
    stamp_img: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    bh, bw = background_img.shape[:2]
    stamp = resize_stamp_for_bg(stamp_img, bw, bh)
    sh, sw = stamp.shape[:2]
    x, y = place_by_orientation(bw, bh, sw, sh)
    canvas = background_img.copy()
    canvas[y:y+sh, x:x+sw] = stamp
    label = np.array([0, (x + sw / 2) / bw, (y + sh / 2) / bh, sw / bw, sh / bh])
    return canvas, label, {"stamp_bbox": [x, y, sw, sh], "background_shape": background_img.shape}


def generate_dataset(output_dir: Path, num_samples: int = 100, dpi: int = 200) -> List[dict]:
    output_dir = Path(output_dir)
    img_dir = output_dir / "images" / "train"
    lbl_dir = output_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    metadata_list = []
    for i in range(num_samples):
        img, label, meta = generate_synthetic_image(dpi=dpi)

        img_path = img_dir / f"synth_{i:04d}.png"
        cv2.imwrite(str(img_path), img)

        lbl_path = lbl_dir / f"synth_{i:04d}.txt"
        with open(lbl_path, "w") as f:
            f.write(f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

        metadata_list.append({**meta, "image": img_path.name})

    return metadata_list



