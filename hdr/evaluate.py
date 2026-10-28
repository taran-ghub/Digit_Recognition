"""Held-out metrics, synthetic strings, and optional independently collected CSV."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .inference import Recognizer


def distance(a, b):
    row = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        new = [i]
        for j, cb in enumerate(b, 1):
            new.append(min(new[-1]+1, row[j]+1, row[j-1]+(ca != cb)))
        row = new
    return row[-1]


def string_metrics(recognizer, samples):
    correct = accepted = edits = chars = total = negative_count = false_accepts = 0
    sequence_correct = sequence_total = agreement_count = 0
    latency, records = [], []
    # Load/warm TensorFlow outside timed calls.
    recognizer.digit(np.zeros((1,28,28,1),np.float32), training=False)
    recognizer.validator(np.zeros((1,28,28,1),np.float32), training=False)
    recognizer.character(np.zeros((1,28,28,1),np.uint8), training=False)
    for image, truth, label in samples:
        start = time.perf_counter()
        result = recognizer.predict(image)
        latency.append((time.perf_counter()-start)*1000)
        pred = result['number']
        if truth is None:
            negative_count += 1
            false_accepts += int(result['accepted'])
        else:
            total += 1
            if result.get('sequence_available'):
                sequence_total += 1
                sequence_correct += int(result.get('sequence_reading') == truth)
                agreement_count += int(result.get('sequence_reading') == result.get('candidate_number'))
            correct += int(pred == truth)
            accepted += int(result['accepted'])
            edits += distance(truth, pred or '')
            chars += len(truth)
        records.append({'sample':label, 'truth':truth, 'prediction':pred,
                        'component_candidate':result.get('candidate_number'),
                        'sequence_reading':result.get('sequence_reading'), 'reason':result['reason']})
    return {'genuine_strings':total, 'exact_string_accuracy':correct/total if total else None,
            'character_error_rate':edits/chars if chars else None,
            'coverage':accepted/total if total else None,
            'selective_exact_accuracy':correct/accepted if accepted else None,
            'negative_inputs':negative_count,
            'negative_false_accept_rate':false_accepts/negative_count if negative_count else None,
            'median_ms':float(np.median(latency)) if latency else None,
            'p95_ms':float(np.percentile(latency,95)) if latency else None,
            'sequence_available':bool(sequence_total),
            'sequence_exact_accuracy':sequence_correct/sequence_total if sequence_total else None,
            'path_agreement_rate':agreement_count/sequence_total if sequence_total else None,
            'records':records}


def synthetic_samples(raw, labels, count):
    rng = np.random.default_rng(2026)
    for i in range(count):
        length = 2 + i % 5
        ids = rng.integers(0,len(raw),length)
        gap = int(rng.integers(4,12))
        canvas = np.zeros((40, 12+length*(28+gap)),np.uint8)
        for j, idx in enumerate(ids):
            y = int(rng.integers(4,9)); x = 6+j*(28+gap)
            canvas[y:y+28,x:x+28] = raw[idx]
        yield Image.fromarray(canvas), ''.join(str(int(labels[k])) for k in ids), f'synthetic-{i}'


def local_samples(csv_path):
    path = Path(csv_path)
    with path.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            target = row['label']
            if target != 'NON_DIGIT' and (not target or any(c not in '0123456789' for c in target)):
                raise ValueError('CSV labels must be digit strings or NON_DIGIT.')
            with Image.open(path.parent/row['path']) as im:
                yield im.copy(), None if target == 'NON_DIGIT' else target, row['path']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--models', default='models')
    ap.add_argument('--output', default='reports')
    ap.add_argument('--strings', type=int, default=500)
    ap.add_argument('--local-csv')
    a = ap.parse_args()
    recognizer = Recognizer(a.models)
    test = np.load(Path(a.data)/'test.npz')
    x, y, negatives = test['x'], test['y'], test['negative']
    p = recognizer.digit.predict(x,batch_size=256,verbose=0)
    q = recognizer.validator.predict(x,batch_size=256,verbose=0).ravel()
    pn = recognizer.digit.predict(negatives,batch_size=256,verbose=0)
    qn = recognizer.validator.predict(negatives,batch_size=256,verbose=0).ravel()
    r = recognizer.character.predict(np.rint(x*255).astype(np.uint8),batch_size=256,verbose=0)
    rn = recognizer.character.predict(np.rint(negatives*255).astype(np.uint8),batch_size=256,verbose=0)
    tc, tv = recognizer.thresholds['classifier'], recognizer.thresholds['validator']
    tr = recognizer.thresholds['character']
    character_digit = r[:,:10].sum(1)
    character_negative_digit = rn[:,:10].sum(1)
    keep = (p.max(1)>=tc)&(q>=tv)&(character_digit>=tr)
    accept_negative = (pn.max(1)>=tc)&(qn>=tv)&(character_negative_digit>=tr)
    report = {'thresholds':recognizer.thresholds, 'test_digit_count':len(y),
        'test_negative_count':len(negatives), 'digit_accuracy':float(accuracy_score(y,p.argmax(1))),
        'digit_classification_report':classification_report(y,p.argmax(1),output_dict=True,zero_division=0),
        'confusion_matrix':confusion_matrix(y,p.argmax(1),labels=np.arange(10)).tolist(),
        'validator_report':classification_report(np.r_[np.ones(len(q)),np.zeros(len(qn))],
            np.r_[q>=tv, qn>=tv], output_dict=True,zero_division=0),
        'character_gate_report':classification_report(
            np.r_[np.ones(len(character_digit)),np.zeros(len(character_negative_digit))],
            np.r_[character_digit>=tr,character_negative_digit>=tr],
            output_dict=True,zero_division=0),
        'pipeline_digit_coverage':float(keep.mean()),
        'pipeline_digit_false_reject_rate':float(1-keep.mean()),
        'pipeline_negative_false_accept_rate':float(accept_negative.mean()),
        'pipeline_selective_digit_accuracy':float((p.argmax(1)[keep]==y[keep]).mean()) if keep.any() else None,
        'confidence_only_at_same_classifier_threshold':{
            'digit_coverage':float((p.max(1)>=tc).mean()),
            'negative_false_accept_rate':float((pn.max(1)>=tc).mean())},
        'synthetic_strings':string_metrics(recognizer,synthetic_samples(test['raw'],y,a.strings))}
    if a.local_csv:
        report['local_inputs'] = string_metrics(recognizer,local_samples(a.local_csv))
    out = Path(a.output); out.mkdir(parents=True,exist_ok=True)
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    print('Evaluation saved:',out/'evaluation.json')
    print('Digit accuracy:',report['digit_accuracy'])
    print('Negative false acceptance:',report['pipeline_negative_false_accept_rate'])
    print('Synthetic exact string accuracy:',report['synthetic_strings']['exact_string_accuracy'])


if __name__ == '__main__':
    main()
