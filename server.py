"""Local FastAPI service for the React DigitLab Studio interface."""
import csv
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from hdr.inference import Recognizer
from hdr.validators import FORMATS, validate_format

ROOT = Path(__file__).resolve().parent
app = FastAPI(title='DigitLab Studio API', version='1.0')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173'],
                   allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
recognizer = None


def get_recognizer():
    global recognizer
    if recognizer is None:
        recognizer = Recognizer(ROOT/'models')
    return recognizer


async def read_image(file):
    content = await file.read()
    if not content or len(content) > 12_000_000:
        raise HTTPException(400, 'Choose a non-empty image smaller than 12 MB.')
    try:
        with Image.open(io.BytesIO(content)) as opened:
            if opened.width * opened.height > 16_000_000:
                raise HTTPException(400, 'Use an image smaller than 16 megapixels.')
            return opened.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, f'Could not read the image: {exc}') from exc


def prepare_camera_image(image):
    """Make a webcam frame resemble the clean, high-contrast training input.

    Webcam frames commonly include desk/background texture and uneven lighting.
    The centre band is the guided reading area in the UI; adaptive thresholding
    retains dark pen strokes even under shadows.
    """
    gray = np.array(image.convert('L'))
    height, width = gray.shape
    # Keep a broad centred horizontal writing zone.  This intentionally avoids
    # room/background detail near the camera-frame borders.
    x_pad, y_top, y_bottom = round(width * .08), round(height * .22), round(height * .78)
    zone = gray[y_top:y_bottom, x_pad:width-x_pad]
    if zone.size == 0:
        return image
    if max(zone.shape) > 1280:
        scale = 1280 / max(zone.shape)
        zone = cv2.resize(zone, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    zone = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(zone)
    clean = cv2.adaptiveThreshold(zone, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 41, 10)
    return Image.fromarray(clean, mode='L')


def serialise_prediction(image, format_name):
    if format_name not in FORMATS:
        raise HTTPException(400, 'Unknown input format.')
    result = get_recognizer().predict(image)
    format_result = validate_format(result['number'], format_name)
    decision = 'accepted' if result['accepted'] and format_result['valid'] else (
        'format_warning' if result['accepted'] else 'rejected')
    return {
        'decision': decision,
        'decision_mode': result.get('decision_mode', 'rejected'),
        'reading': result.get('number'),
        'candidate_reading': result.get('candidate_number'),
        'reason': result.get('reason'),
        'format': format_result,
        'components': result.get('components', []),
        'boxes': [list(box) for box in result.get('boxes', [])],
        'sequence_available': result.get('sequence_available', False),
        'sequence_reading': result.get('sequence_reading'),
        'sequence_confidence': result.get('sequence_confidence'),
        'thresholds': get_recognizer().thresholds,
    }


@app.get('/api/health')
def health():
    model_dir = ROOT/'models'
    return {'ready': all((model_dir/name).exists() for name in
            ('digit.keras', 'validator.keras', 'character.keras', 'thresholds.json')),
            'sequence_ready': (model_dir/'sequence.keras').exists(),
            'formats': {name: detail[0] for name, detail in FORMATS.items()}}


@app.post('/api/predict')
async def predict(file: UploadFile = File(...), format_name: str = Form('General number'),
                  source: str = Form('web')):
    image = await read_image(file)
    if source == 'camera':
        image = prepare_camera_image(image)
    return serialise_prediction(image, format_name)


@app.post('/api/feedback')
async def feedback(file: UploadFile = File(...), actual_label: str = Form(...),
                   format_name: str = Form('General number'), source: str = Form('web')):
    actual = actual_label.strip()
    if actual != 'NON_DIGIT' and not actual.isdigit():
        raise HTTPException(400, 'Use digits only, or exactly NON_DIGIT.')
    image = await read_image(file)
    result = serialise_prediction(image, format_name)
    folder = ROOT/'feedback'; folder.mkdir(exist_ok=True)
    sample_id = uuid.uuid4().hex
    image_path = folder/f'{sample_id}.png'; image.convert('RGB').save(image_path)
    table = folder/'feedback.csv'
    row = {'sample_id': sample_id, 'created_utc': datetime.now(timezone.utc).isoformat(),
           'source': source, 'actual_label': actual, 'format': format_name,
           'decision': result['decision'], 'reading': result['reading'],
           'candidate_reading': result['candidate_reading'],
           'sequence_reading': result['sequence_reading'], 'reason': result['reason'],
           'image': image_path.name}
    new = not table.exists()
    with table.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if new:
            writer.writeheader()
        writer.writerow(row)
    return {'saved': True, 'sample_id': sample_id}


@app.get('/api/dashboard')
def dashboard():
    report_path = ROOT/'reports'/'evaluation.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    feedback_path = ROOT/'feedback'/'feedback.csv'
    feedback_count = 0
    if feedback_path.exists():
        with feedback_path.open(encoding='utf-8') as handle:
            feedback_count = sum(1 for _ in csv.DictReader(handle))
    return {'report': report, 'feedback_count': feedback_count}


DIST = ROOT/'frontend'/'dist'
if DIST.exists():
    app.mount('/', StaticFiles(directory=DIST, html=True), name='frontend')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='127.0.0.1', port=8000, reload=False)
