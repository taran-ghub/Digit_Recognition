"""Download official datasets; split source images before any composition."""
import argparse
import json
import gzip
import hashlib
import struct
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from .preprocess import preprocess_many


def generated_negatives(count, seed):
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(count):
        im = Image.new('L', (28, 28), 0)
        d = ImageDraw.Draw(im)
        kind = rng.integers(0, 4)
        width = int(rng.integers(1, 4))
        if kind == 0:
            d.line((5, 14, 23, 14), fill=255, width=width)
            d.line((14, 5, 14, 23), fill=255, width=width)
        elif kind == 1:
            for y in (10, 18):
                d.line((5, y, 23, y), fill=255, width=width)
        elif kind == 2:
            points = [tuple(map(int, p)) for p in rng.integers(3, 25, (7, 2))]
            d.line(points, fill=255, width=width)
        else:
            for x, y in rng.integers(3, 25, (20, 2)):
                d.ellipse((int(x), int(y), int(x)+1, int(y)+1), fill=255)
        images.append(np.array(im))
    return preprocess_many(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data')
    parser.add_argument('--emnist-zip', help='Path to an already downloaded official EMNIST gzip.zip')
    args = parser.parse_args()
    import tensorflow as tf
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (images, labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    tr, va = train_test_split(np.arange(len(labels)), test_size=10000, random_state=42, stratify=labels)
    print('Loading EMNIST Letters (first download can be large).', flush=True)
    expected_hash = 'fb9bb67e33772a9cc0b895e4ecf36d2cf35be8b709693c3564cea2a019fcda8e'
    if args.emnist_zip:
        archive = Path(args.emnist_zip)
        with archive.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != expected_hash:
            raise ValueError('Archive checksum differs from the official EMNIST gzip.zip. Check that the download is complete.')
    else:
        try:
            archive = tf.keras.utils.get_file(
                'emnist-gzip.zip', 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip',
                file_hash=expected_hash, hash_algorithm='sha256')
        except Exception as exc:
            raise SystemExit(
                'EMNIST download failed. Download the Binary/MNIST-format archive from '
                'https://www.nist.gov/itl/products-and-services/emnist-dataset '
                'and run: python -m hdr.prepare --emnist-zip "C:/path/to/gzip.zip". '
                f'Original error: {exc}') from exc
    emnist = {}
    with zipfile.ZipFile(archive) as zipped:
        for split in ('train', 'test'):
            prefix = f'gzip/emnist-letters-{split}'
            raw_images = gzip.decompress(zipped.read(prefix+'-images-idx3-ubyte.gz'))
            magic, count, height, width = struct.unpack('>IIII', raw_images[:16])
            raw_labels = gzip.decompress(zipped.read(prefix+'-labels-idx1-ubyte.gz'))
            label_magic, label_count = struct.unpack('>II', raw_labels[:8])
            if magic != 2051 or label_magic != 2049 or (height,width)!=(28,28) or count!=label_count:
                raise ValueError('Invalid EMNIST IDX header.')
            ex = np.frombuffer(raw_images, dtype=np.uint8, offset=16).reshape(count,28,28)
            ey = np.frombuffer(raw_labels, dtype=np.uint8, offset=8).copy()
            if len(ey)!=count or ey.min()!=1 or ey.max()!=26:
                raise ValueError('Expected EMNIST Letters labels 1 through 26.')
            # Raw EMNIST images are transposed relative to upright handwriting.
            emnist[split] = (ex.transpose(0,2,1).copy(), ey)
    etr, eva = train_test_split(np.arange(len(emnist['train'][1])), test_size=.1,
                               random_state=42, stratify=emnist['train'][1])
    partitions = {'train': (images[tr], labels[tr], emnist['train'][0][etr]),
                  'val': (images[va], labels[va], emnist['train'][0][eva]),
                  'test': (test_images, test_labels, emnist['test'][0])}
    for index, (name, (x, y, letters)) in enumerate(partitions.items()):
        print('Preprocessing', name, flush=True)
        # Balance binary data: 80% letter negatives, 20% generated negatives.
        rng = np.random.default_rng(42+index)
        letter_count = min(len(letters), int(len(x)*.8))
        selected = rng.choice(len(letters), letter_count, replace=False)
        negatives = np.concatenate([preprocess_many(letters[selected]),
                                   generated_negatives(len(x)-letter_count, 100+index)])
        np.savez_compressed(out/f'{name}.npz', x=preprocess_many(x), y=y,
                            negative=negatives, raw=x, negative_letter_indices=selected)
    np.savez(out/'split_ids.npz', mnist_train=tr, mnist_val=va,
             emnist_train=etr, emnist_val=eva)
    (out/'manifest.json').write_text(json.dumps({
        'seed':42, 'mnist':'official 60000 train -> 50000 train + 10000 validation; official 10000 test',
        'emnist':'letters; official train -> 90/10 split; official test untouched',
        'negative_mix':'80% letters + 20% procedural negatives where enough letters available',
        'preprocessing':'border polarity, tight crop, fit 20x20, pad 28x28, divide by255',
        'synthetic_negative_limitation':'Different seeds but same shape families across splits; not unseen-family validation.',
        'local_handwriting':'Not collected; report separately when available.'}, indent=2))
    print('Datasets ready in', out)


if __name__ == '__main__':
    main()
