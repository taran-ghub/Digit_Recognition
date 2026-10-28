import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from .models import build_model


def scores(model, x):
    return model.predict(x, batch_size=256, verbose=0)


def calibrate(p, q, pn, qn, target=.05):
    # Maximize genuine-digit retention subject to validation negative FAR <=1%.
    best = None
    for tc in np.linspace(.5, 1, 51):
        for tv in np.linspace(.5, 1, 51):
            retention = float(np.mean((p.max(1) >= tc) & (q >= tv)))
            far = float(np.mean((pn.max(1) >= tc) & (qn >= tv)))
            if far <= target and (best is None or (retention, -far) > best[0]):
                best = ((retention, -far), float(tc), float(tv))
    if best is None or best[0][0] == 0:
        raise RuntimeError('No useful thresholds meet validation FAR target. Inspect data/model quality; do not use test data to tune.')
    return {'classifier':best[1], 'validator':best[2], 'validation_digit_retention':best[0][0],
            'validation_negative_false_accept_rate':-best[0][1], 'target_false_accept_rate':target}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--models', default='models')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--limit', type=int, help='Smoke training only: cap samples; results are not final.')
    a = ap.parse_args()
    tf.keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    directory = Path(a.models)
    directory.mkdir(parents=True, exist_ok=True)
    # Invalidate prior calibration before replacing either model.
    (directory/'thresholds.json').unlink(missing_ok=True)
    train = np.load(Path(a.data)/'train.npz')
    val = np.load(Path(a.data)/'val.npz')
    n = a.limit or len(train['x'])
    x, y, neg = train['x'][:n], train['y'][:n], train['negative'][:n]
    digit = build_model()
    validator = build_model(binary=True)
    histories = {}
    for name, model, tx, ty, vx, vy in [
        ('digit', digit, x, y, val['x'], val['y']),
        ('validator', validator, np.concatenate([x, neg]), np.r_[np.ones(len(x)), np.zeros(len(neg))],
         np.concatenate([val['x'], val['negative']]), np.r_[np.ones(len(val['x'])), np.zeros(len(val['negative']))])]:
        history = model.fit(tx, ty, validation_data=(vx, vy), batch_size=64,
                            epochs=a.epochs, shuffle=True, callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])
        model.save(directory/f'{name}.keras')
        histories[name] = history.history
    thresholds = calibrate(scores(digit,val['x']), scores(validator,val['x']).ravel(),
                           scores(digit,val['negative']), scores(validator,val['negative']).ravel())
    (directory/'thresholds.json').write_text(json.dumps(thresholds,indent=2))
    (directory/'training.json').write_text(json.dumps({'seed':42,'epochs_requested':a.epochs,
        'smoke_training':a.limit is not None, 'history':histories,
        'tensorflow':tf.__version__, 'classes':list(range(10))},indent=2))
    print('Models and calibrated thresholds saved to', directory)


if __name__ == '__main__':
    main()
