"""Train and calibrate the 36-class digit/letter discriminator."""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .models import build_character_model


def digit_mass(model, x):
    return model.predict(x, batch_size=256, verbose=0)[:, :10].sum(axis=1)


def calibrate(digit_scores, letter_scores, target=.05):
    best = None
    for threshold in np.linspace(.05, 1.0, 191):
        retention = float(np.mean(digit_scores >= threshold))
        false_acceptance = float(np.mean(letter_scores >= threshold))
        if false_acceptance <= target and (best is None or retention > best[0]):
            best = (retention, float(threshold), false_acceptance)
    if best is None:
        raise RuntimeError('No character threshold meets the validation target.')
    return {'character': best[1], 'character_validation_digit_retention': best[0],
            'character_validation_letter_false_accept_rate': best[2],
            'character_target_letter_false_accept_rate': target}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--models', default='models')
    ap.add_argument('--epochs', type=int, default=20)
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(42)
    train = np.load(Path(args.data)/'character_train.npz')
    val = np.load(Path(args.data)/'character_val.npz')
    model = build_character_model()
    history = model.fit(train['x'], train['y'], validation_data=(val['x'], val['y']),
                        batch_size=64, epochs=args.epochs, shuffle=True,
                        callbacks=[tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss', patience=3, restore_best_weights=True)])
    model_dir = Path(args.models); model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir/'character.keras')
    digit = val['y'] < 10
    scores = digit_mass(model, val['x'])
    calibrated = calibrate(scores[digit], scores[~digit])
    threshold_path = model_dir/'thresholds.json'
    thresholds = json.loads(threshold_path.read_text())
    thresholds.update(calibrated)
    threshold_path.write_text(json.dumps(thresholds, indent=2))
    (model_dir/'character_training.json').write_text(json.dumps({
        'seed': 42, 'history': history.history, 'tensorflow': tf.__version__,
        'classes': {'0-9': 'digits', '10-35': 'A-Z case merged'}}, indent=2))
    print('Character model and threshold saved to', model_dir)
    print(calibrated)


if __name__ == '__main__':
    main()
