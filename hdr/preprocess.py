"""Shared preprocessing: dark/light backgrounds, aspect ratio, and segmentation."""
import cv2
import numpy as np
from PIL import Image, ImageOps


def foreground(image):
    if isinstance(image, Image.Image):
        image = ImageOps.exif_transpose(image)
        if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
            rgba = image.convert('RGBA')
            background = Image.new('RGBA', rgba.size, 'white')
            image = Image.alpha_composite(background, rgba)
        gray = np.array(image.convert('L'))
    else:
        gray = np.asarray(image, dtype=np.uint8)
        if gray.ndim != 2:
            raise ValueError('Expected grayscale array or PIL image.')
    if gray.size == 0:
        raise ValueError('Empty image.')
    if max(gray.shape) > 1600:
        ratio = 1600 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    # Estimate background from the border, not average foreground intensity.
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    ink = 255 - gray if np.median(border) > 127 else gray.copy()
    if int(ink.max()) - int(ink.min()) < 15:
        return np.zeros_like(ink), np.zeros_like(ink, dtype=bool)
    _, mask = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ink, mask > 0


def normalize(ink, mask=None):
    """Tight crop -> fit 20x20 -> centered 28x28, values in [0,1]."""
    if mask is None:
        _, mask = foreground(ink)
    ys, xs = np.where(mask)
    canvas = np.zeros((28, 28), np.float32)
    if len(xs) == 0:
        return canvas[..., None]
    crop = ink[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = crop.shape
    scale = 20 / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    # Very thin high-resolution strokes can become faint after downscaling.
    # Normal MNIST-like crops peak near 255, so only restore unusually faint ink.
    if 0 < float(crop.max()) < 190:
        crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)
        crop = cv2.dilate(crop, np.ones((2, 2), np.uint8), iterations=1)
    x, y = (28-nw)//2, (28-nh)//2
    canvas[y:y+nh, x:x+nw] = crop.astype(np.float32) / 255
    return canvas[..., None]


def preprocess_many(images):
    result = []
    for image in images:
        ink, mask = foreground(np.asarray(image).squeeze())
        result.append(normalize(ink, mask))
    return np.stack(result)


def segment(image, max_digits=12):
    ink, mask = foreground(image)
    if mask.sum() < 3:
        return ink, [], [], 'Blank or insufficient ink.'
    # Projection groups retain small marks; silently discarding a decimal point
    # would incorrectly turn 1.2 into 12. No unconditional small-blob removal.
    occupied = np.flatnonzero(mask.any(axis=0))
    runs = np.split(occupied, np.where(np.diff(occupied) > 1)[0] + 1)
    boxes, crops = [], []
    for columns in runs:
        x1, x2 = int(columns[0]), int(columns[-1])+1
        rows = np.flatnonzero(mask[:, x1:x2].any(axis=1))
        y1, y2 = int(rows[0]), int(rows[-1])+1
        boxes.append((x1, y1, x2, y2))
        crops.append(normalize(ink[y1:y2, x1:x2], mask[y1:y2, x1:x2]))
    if len(crops) > max_digits:
        return ink, boxes, crops, 'Too many components; use one clean row of digits.'
    heights = np.array([b[3]-b[1] for b in boxes])
    typical = heights.max()
    if np.any(heights < typical * .25):
        return ink, boxes, crops, 'Small detached mark or noise: input needs review.'
    # Conservative alignment check; cannot detect all multi-line/touching cases.
    if max(b[1] for b in boxes) >= min(b[3] for b in boxes):
        return ink, boxes, crops, 'Components are not aligned in one row.'
    if any((b[2]-b[0]) > 1.6*(b[3]-b[1]) for b in boxes):
        return ink, boxes, crops, 'Wide component: possibly touching digits or a symbol.'
    return ink, boxes, crops, None
