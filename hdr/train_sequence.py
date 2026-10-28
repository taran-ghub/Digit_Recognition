"""Train the end-to-end CNN–BiLSTM–CTC number recognizer on synthetic MNIST rows."""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .models import build_sequence_model
from .sequence import MAX_LENGTH, make_batch, decode_logits


def ctc_loss(labels, lengths, logits):
    batch = tf.shape(logits)[0]
    times = tf.fill([batch], tf.shape(logits)[1])
    loss = tf.nn.ctc_loss(labels=labels, logits=logits, label_length=lengths,
                          logit_length=times, logits_time_major=False, blank_index=10)
    return tf.reduce_mean(loss)


def evaluate(model, raw, labels, seed, batches=20, batch_size=64):
    rng = np.random.default_rng(seed)
    correct = characters = edits = total = 0
    for _ in range(batches):
        x, y, lengths = make_batch(raw, labels, rng, batch_size)
        predictions, _ = decode_logits(model(x, training=False).numpy())
        for predicted, target, size in zip(predictions, y, lengths):
            truth = ''.join(map(str, target[:size]))
            correct += predicted == truth; total += 1; characters += len(truth)
            previous = list(range(len(predicted)+1))
            for i, a in enumerate(truth, 1):
                row = [i]
                for j, b in enumerate(predicted, 1):
                    row.append(min(row[-1]+1, previous[j]+1, previous[j-1]+(a != b)))
                previous = row
            edits += previous[-1]
    return {'exact_string_accuracy': correct/total, 'character_error_rate': edits/characters,
            'samples': total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--models', default='models')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--steps-per-epoch', type=int, default=450)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--validation-batches', type=int, default=25)
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(2026)
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    ids = np.load(Path(args.data)/'split_ids.npz')
    train_raw, train_labels = train_x[ids['mnist_train']], train_y[ids['mnist_train']]
    val_raw, val_labels = train_x[ids['mnist_val']], train_y[ids['mnist_val']]
    model = build_sequence_model()
    optimizer = tf.keras.optimizers.Adam(0.001)
    rng = np.random.default_rng(2026)
    best, stale, history = -1.0, 0, []
    model_dir = Path(args.models); model_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        losses = []
        for _ in range(args.steps_per_epoch):
            x, y, lengths = make_batch(train_raw, train_labels, rng, args.batch_size)
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = ctc_loss(y, lengths, logits)
            optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
            losses.append(float(loss))
        validation = evaluate(model, val_raw, val_labels, 5000 + epoch,
                              args.validation_batches, args.batch_size)
        record = {'epoch': epoch, 'loss': float(np.mean(losses)), **validation}; history.append(record)
        print(record, flush=True)
        if validation['exact_string_accuracy'] > best:
            best, stale = validation['exact_string_accuracy'], 0
            model.save(model_dir/'sequence.keras')
        else:
            stale += 1
            if stale >= 3:
                print('Early stopping after three non-improving validation epochs.')
                break
    model = tf.keras.models.load_model(model_dir/'sequence.keras', compile=False)
    held_out = evaluate(model, test_x, test_y, 9001, 50, args.batch_size)
    (model_dir/'sequence_training.json').write_text(json.dumps({
        'architecture': 'CNN–BiLSTM–CTC, 10 digits plus CTC blank',
        'seed': 2026, 'epochs_requested': args.epochs, 'history': history,
        'held_out_synthetic_mnist_strings': held_out, 'max_length': MAX_LENGTH}, indent=2))
    print('Sequence model saved to', model_dir)
    print('Held-out synthetic sequence metrics:', held_out)


if __name__ == '__main__':
    main()
