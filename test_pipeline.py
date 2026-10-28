import numpy as np
from PIL import Image, ImageDraw
from hdr.preprocess import foreground, normalize, segment
from hdr.inference import decide
from hdr.evaluate import distance
from hdr.sequence import decode_logits
from hdr.validators import validate_format


def sample():
    im = Image.new('L',(100,50),255)
    d = ImageDraw.Draw(im)
    d.rectangle((10,10,13,40),fill=0)
    d.ellipse((35,10,55,40),outline=0,width=3)
    return im


def test_blank_both_polarities():
    for color in (0,128,255):
        assert segment(Image.new('L',(100,50),color))[3] is not None


def test_polarity_and_geometry():
    im = sample()
    a, boxes, crops, error = segment(im)
    _, boxes2, crops2, error2 = segment(Image.fromarray(255-np.array(im)))
    assert error is None and error2 is None
    assert len(boxes)==2 and boxes[0][0]<boxes[1][0]
    assert boxes==boxes2
    np.testing.assert_allclose(crops,crops2)
    assert crops[0].shape==(28,28,1)
    assert 0<=crops[0].min()<=crops[0].max()<=1
    assert np.count_nonzero(crops[0][...,0].any(axis=0)) < 5  # thin 1 retained


def test_leading_zeros_and_whole_string_rejection():
    p = np.eye(10)[[0,0,7]]
    threshold = {'classifier':.8,'validator':.8}
    assert decide(p,[.9,.9,.9],threshold)['number']=='007'
    bad = decide(p,[.9,.2,.9],threshold)
    assert bad['number'] is None and bad['candidate_number']=='007' and not bad['accepted']


def test_character_model_can_reject_a_letter():
    p = np.eye(10)[[3, 2]]
    r = np.zeros((2, 36)); r[0, 3] = 1; r[1, 14] = 1  # digit 3, letter E
    result = decide(p, [.99, .99], {'classifier':.8, 'validator':.8, 'character':.5}, r)
    assert result['candidate_number'] == '32'
    assert result['components'][0]['accepted']
    assert not result['components'][1]['accepted']
    assert result['components'][1]['character_class'] == 'E'


def test_sequence_recovery_is_only_eligible_for_soft_valid_digits():
    p = np.eye(10)[[4, 2]]
    result = decide(p, [.70, .70],
                    {'classifier': .8, 'validator': .98, 'validator_recovery': .55})
    assert not result['accepted']
    assert result['recovery_eligible']


def test_faint_thin_strokes_are_restored():
    im = np.zeros((300, 300), np.uint8)
    cv2 = __import__('cv2')
    cv2.line(im, (30, 30), (270, 270), 80, 1)
    result = normalize(im, im > 0)[..., 0]
    assert result.max() > .95


def test_decimal_mark_not_silently_removed():
    im = sample(); ImageDraw.Draw(im).ellipse((25,38,27,40),fill=0)
    assert segment(im)[3] is not None


def test_transparent_background():
    im = sample().convert('RGBA')
    pixels = np.array(im); pixels[pixels[...,0]==255,3]=0
    assert len(segment(Image.fromarray(pixels))[2])==2


def test_edit_distance():
    assert distance('007','07')==1
    assert distance('123','')==3
    assert distance('123','143')==1


def test_ctc_greedy_decode_collapses_repeated_frames_and_blanks():
    logits = np.full((1, 7, 11), -9., np.float32)
    # 1, 1, blank, 2, 2, blank, 0 should read 120.
    logits[0, np.arange(7), [1, 1, 10, 2, 2, 10, 0]] = 9
    readings, confidence = decode_logits(logits)
    assert readings == ['120'] and confidence[0] > .99


def test_transparent_format_rules():
    assert validate_format('9876543210', 'Indian mobile number')['valid']
    assert not validate_format('4876543210', 'Indian mobile number')['valid']
    assert validate_format('160001', 'Indian PIN code')['valid']
    assert not validate_format('960001', 'Indian PIN code')['valid']
