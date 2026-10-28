"""Synthetic full-number data and CRNN/CTC decoding utilities."""
import cv2
import numpy as np
from PIL import Image

from .preprocess import foreground

HEIGHT, WIDTH, MAX_LENGTH, BLANK = 48, 192, 10, 10


def preprocess_sequence(image):
    """Convert one complete handwritten row to the fixed CRNN input shape."""
    ink, mask = foreground(image)
    canvas = np.zeros((HEIGHT, WIDTH), np.float32)
    ys, xs = np.where(mask)
    if not len(xs):
        return canvas[..., None]
    crop = ink[ys.min():ys.max()+1, xs.min():xs.max()+1]
    scale = min((HEIGHT - 8) / crop.shape[0], (WIDTH - 8) / crop.shape[1])
    size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    if 0 < float(crop.max()) < 190:
        crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)
        crop = cv2.dilate(crop, np.ones((2, 2), np.uint8), iterations=1)
    x = (WIDTH - crop.shape[1]) // 2
    y = (HEIGHT - crop.shape[0]) // 2
    canvas[y:y+crop.shape[0], x:x+crop.shape[1]] = crop / 255.0
    return canvas[..., None]


def make_number_image(raw_digits, labels, rng, min_length=1, max_length=MAX_LENGTH):
    """Create a varied one-line number from source digit images, without label leakage."""
    length = int(rng.integers(min_length, max_length + 1))
    ids = rng.integers(0, len(raw_digits), length)
    # Wide enough for a ten-digit phone number before it is rescaled to WIDTH.
    canvas = np.zeros((72, 480), np.uint8)
    cursor = int(rng.integers(5, 14))
    placed = []
    for index in ids:
        glyph = raw_digits[index]
        scale = float(rng.uniform(.75, 1.18))
        glyph = cv2.resize(glyph, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        angle = float(rng.uniform(-12, 12))
        matrix = cv2.getRotationMatrix2D((glyph.shape[1]/2, glyph.shape[0]/2), angle, 1)
        glyph = cv2.warpAffine(glyph, matrix, (glyph.shape[1], glyph.shape[0]), borderValue=0)
        if cursor + glyph.shape[1] + 2 >= canvas.shape[1]:
            break
        y = int(rng.integers(7, max(8, canvas.shape[0] - glyph.shape[0] - 4)))
        target = canvas[y:y+glyph.shape[0], cursor:cursor+glyph.shape[1]]
        np.maximum(target, glyph, out=target)
        placed.append(index)
        cursor += glyph.shape[1] + int(rng.integers(-2, 9))
    chosen = np.asarray([labels[i] for i in placed], np.int32)
    # The generous canvas prevents clipping; preprocessing handles final scaling.
    return preprocess_sequence(Image.fromarray(canvas)), chosen


def make_batch(raw_digits, labels, rng, batch_size, min_length=1, max_length=MAX_LENGTH):
    x = np.empty((batch_size, HEIGHT, WIDTH, 1), np.float32)
    y = np.zeros((batch_size, MAX_LENGTH), np.int32)
    lengths = np.empty(batch_size, np.int32)
    for i in range(batch_size):
        x[i], seq = make_number_image(raw_digits, labels, rng, min_length, max_length)
        y[i, :len(seq)] = seq
        lengths[i] = len(seq)
    return x, y, lengths


def decode_logits(logits):
    """Greedy CTC decoding; returns strings and a simple frame-confidence diagnostic."""
    logits = np.asarray(logits)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted); probabilities /= probabilities.sum(axis=-1, keepdims=True)
    paths = probabilities.argmax(axis=-1)
    readings, confidence = [], []
    for path, probability in zip(paths, probabilities):
        ids, previous = [], BLANK
        for item in path:
            item = int(item)
            if item != BLANK and item != previous:
                ids.append(item)
            previous = item
        readings.append(''.join(map(str, ids)))
        confidence.append(float(probability.max(axis=-1).mean()))
    return readings, confidence
