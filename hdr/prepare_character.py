"""Build uint8 datasets for the 10-digit + 26-letter character model."""
import argparse
import gzip
import hashlib
import struct
import zipfile
from pathlib import Path

import numpy as np

from .preprocess import preprocess_many

EXPECTED_HASH = 'fb9bb67e33772a9cc0b895e4ecf36d2cf35be8b709693c3564cea2a019fcda8e'


def read_letters(archive, split):
    prefix = f'gzip/emnist-letters-{split}'
    with zipfile.ZipFile(archive) as zipped:
        raw_images = gzip.decompress(zipped.read(prefix + '-images-idx3-ubyte.gz'))
        raw_labels = gzip.decompress(zipped.read(prefix + '-labels-idx1-ubyte.gz'))
    magic, count, height, width = struct.unpack('>IIII', raw_images[:16])
    label_magic, label_count = struct.unpack('>II', raw_labels[:8])
    if magic != 2051 or label_magic != 2049 or (height, width) != (28, 28) or count != label_count:
        raise ValueError('Invalid EMNIST IDX header.')
    images = np.frombuffer(raw_images, np.uint8, offset=16).reshape(count, 28, 28)
    labels = np.frombuffer(raw_labels, np.uint8, offset=8).copy()
    if labels.min() != 1 or labels.max() != 26:
        raise ValueError('Expected EMNIST Letters labels 1 through 26.')
    return images.transpose(0, 2, 1).copy(), labels


def as_uint8(images):
    return np.rint(preprocess_many(images) * 255).astype(np.uint8)


def save_set(path, digit_x, digit_y, letter_x, letter_y, indices):
    lx = as_uint8(letter_x[indices])
    ly = 10 + letter_y[indices].astype(np.int64) - 1
    dx = np.rint(digit_x * 255).astype(np.uint8)
    x = np.concatenate([dx, lx])
    y = np.concatenate([digit_y.astype(np.int64), ly])
    order = np.random.default_rng(42).permutation(len(y))
    np.savez_compressed(path, x=x[order], y=y[order])
    print(path.name, x.shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emnist-zip', required=True)
    ap.add_argument('--data', default='data')
    ap.add_argument('--letters-per-class', type=int, default=2000)
    args = ap.parse_args()
    archive, data = Path(args.emnist_zip), Path(args.data)
    with archive.open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != EXPECTED_HASH:
            raise ValueError('Archive checksum differs from official EMNIST gzip.zip.')
    splits = np.load(data/'split_ids.npz')
    train_letters, train_y = read_letters(archive, 'train')
    test_letters, test_y = read_letters(archive, 'test')
    train_ids, val_ids = splits['emnist_train'], splits['emnist_val']
    rng = np.random.default_rng(42)
    selected = []
    for label in range(1, 27):
        available = train_ids[train_y[train_ids] == label]
        selected.extend(rng.choice(available, min(args.letters_per_class, len(available)), replace=False))
    selected = np.asarray(selected)
    for name, letter_images, letter_labels, ids in [
        ('character_train.npz', train_letters, train_y, selected),
        ('character_val.npz', train_letters, train_y, val_ids),
        ('character_test.npz', test_letters, test_y, np.arange(len(test_y)))]:
        digits = np.load(data/name.replace('character_', ''))
        save_set(data/name, digits['x'], digits['y'], letter_images, letter_labels, ids)
    print('Character datasets ready in', data)


if __name__ == '__main__':
    main()
