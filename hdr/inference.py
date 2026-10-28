import json
from pathlib import Path
import numpy as np
from .preprocess import segment
from .sequence import preprocess_sequence, decode_logits


def decide(probabilities, validity, thresholds, character_probabilities=None):
    probabilities, validity = np.asarray(probabilities), np.asarray(validity).reshape(-1)
    if len(probabilities) != len(validity):
        raise ValueError('Mismatched predictions.')
    digits = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    accepted = (validity >= thresholds['validator']) & (confidence >= thresholds['classifier'])
    character_mass = np.ones(len(digits))
    character_class = ['not checked'] * len(digits)
    if character_probabilities is not None:
        character_probabilities = np.asarray(character_probabilities)
        if len(character_probabilities) != len(digits) or character_probabilities.shape[1] != 36:
            raise ValueError('Expected one 36-class character prediction per component.')
        character_mass = character_probabilities[:, :10].sum(axis=1)
        character_ids = character_probabilities.argmax(axis=1)
        character_class = [str(int(i)) if i < 10 else chr(ord('A') + int(i) - 10)
                           for i in character_ids]
        accepted &= character_mass >= thresholds['character']
    # The binary validator is useful for the conservative *strict* path, but
    # it is the least reliable model on thick/freehand canvas strokes.  Do not
    # make its score a second hidden requirement for sequence recovery.
    #
    # Recovery is enabled only when every crop has a very confident digit-CNN
    # prediction, the character model's TOP class is a digit (a letter is a
    # hard veto), and the independently trained CRNN returns the exact same
    # complete string in ``predict`` below.  This is deliberately stronger
    # than merely lowering the validator threshold.
    recovery = confidence >= thresholds.get('sequence_recovery_classifier', .95)
    if character_probabilities is not None:
        recovery &= character_ids < 10
    rows = [dict(digit=str(int(d)), classifier_score=float(c), digit_score=float(v),
                 character_digit_score=float(cm), character_class=cc,
                 accepted=bool(a), recovery_eligible=bool(re))
            for d, c, v, cm, cc, a, re in zip(digits, confidence, validity,
                                               character_mass, character_class, accepted, recovery)]
    ok = bool(len(rows) and accepted.all())
    candidate = ''.join(str(int(d)) for d in digits)
    return dict(accepted=ok, strict_accepted=ok, recovery_eligible=bool(len(rows) and recovery.all()),
                number=candidate if ok else None, candidate_number=candidate,
                decision_mode='strict' if ok else 'rejected',
                reason=None if ok else 'At least one component is uncertain or non-digit.', components=rows)


class Recognizer:
    def __init__(self, model_dir='models'):
        import tensorflow as tf
        directory = Path(model_dir)
        required = ['digit.keras', 'validator.keras', 'character.keras', 'thresholds.json']
        missing = [f for f in required if not (directory/f).exists()]
        if missing:
            raise FileNotFoundError('Train models first. Missing: ' + ', '.join(missing))
        self.digit = tf.keras.models.load_model(directory/'digit.keras', compile=False)
        self.validator = tf.keras.models.load_model(directory/'validator.keras', compile=False)
        self.character = tf.keras.models.load_model(directory/'character.keras', compile=False)
        self.sequence = (tf.keras.models.load_model(directory/'sequence.keras', compile=False)
                         if (directory/'sequence.keras').exists() else None)
        self.thresholds = json.loads((directory/'thresholds.json').read_text())
        # Kept separate from strict thresholds so a clear multi-digit canvas
        # reading can be recovered only through exact dual-path agreement.
        self.thresholds.setdefault('sequence_recovery_classifier', .95)

    def predict(self, image):
        sequence_reading, sequence_confidence = self.predict_sequence(image)
        ink, boxes, crops, error = segment(image)
        if error:
            return dict(accepted=False, number=None, candidate_number=None, reason=error,
                        components=[], boxes=boxes, ink=ink, crops=crops,
                        sequence_reading=sequence_reading, sequence_confidence=sequence_confidence,
                        sequence_available=self.sequence is not None)
        x = np.stack(crops)
        # Direct calls avoid starting a data pipeline for tiny interactive batches.
        p = self.digit(x, training=False).numpy()
        q = self.validator(x, training=False).numpy()
        # Character model owns its uint8 rescaling layer.
        character_x = np.rint(x * 255).astype(np.uint8)
        r = self.character(character_x, training=False).numpy()
        result = decide(p, q, self.thresholds, r)
        result['sequence_reading'] = sequence_reading
        result['sequence_confidence'] = sequence_confidence
        result['sequence_available'] = self.sequence is not None
        if self.sequence is not None:
            agrees = bool(sequence_reading and sequence_reading == result['candidate_number'])
            if result['strict_accepted'] and agrees:
                result['decision_mode'] = 'strict_agreement'
            elif result['recovery_eligible'] and agrees:
                result['accepted'] = True
                result['number'] = result['candidate_number']
                result['reason'] = None
                result['decision_mode'] = 'sequence_supported'
            else:
                result['accepted'] = False
                result['number'] = None
                result['decision_mode'] = 'rejected'
                result['reason'] = ('The component and full-number recognition paths disagree; '
                                    'please rewrite the number.') if not agrees else (
                                    'At least one component is uncertain or non-digit.')
        result.update(boxes=boxes, ink=ink, crops=crops)
        return result

    def predict_sequence(self, image):
        if self.sequence is None:
            return None, None
        x = preprocess_sequence(image)[None, ...]
        logits = self.sequence(x, training=False).numpy()
        readings, confidence = decode_logits(logits)
        return readings[0], confidence[0]
