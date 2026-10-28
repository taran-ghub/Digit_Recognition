"""Run with: python -m streamlit run app.py"""
from pathlib import Path
import csv
import json
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError
from streamlit_drawable_canvas import st_canvas

from hdr.inference import Recognizer
from hdr.validators import FORMATS, validate_format

try:
    from streamlit_webrtc import webrtc_streamer
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='DigitLab | Handwritten Numbers', page_icon='✍️', layout='wide')
st.title('DigitLab')
st.subheader('Handwritten digits, checked before acceptance')
st.write('Draw a number or upload one digit or a single row of clearly separated digits.')
st.caption('Research prototype • Integers only • Touching digits, decimals and multiple lines are outside the supported scope.')


@st.cache_resource
def get_recognizer():
    return Recognizer(ROOT/'models')


def save_feedback(image, result, actual, source, format_name):
    root = ROOT/'feedback'; root.mkdir(exist_ok=True)
    sample_id = uuid.uuid4().hex
    image_path = root/f'{sample_id}.png'
    image.convert('RGB').save(image_path)
    table = root/'feedback.csv'
    row = {'sample_id': sample_id, 'created_utc': datetime.now(timezone.utc).isoformat(),
           'source': source, 'actual_label': actual.strip(), 'format': format_name,
           'accepted': result['accepted'], 'reading': result.get('number'),
           'candidate_reading': result.get('candidate_number'),
           'sequence_reading': result.get('sequence_reading'),
           'reason': result.get('reason'), 'image': image_path.name}
    exists = table.exists()
    with table.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def show_result(image, source_caption, format_name):
    start = time.perf_counter()
    result = recognizer.predict(image)
    elapsed = (time.perf_counter()-start)*1000
    left, right = st.columns(2)
    with left:
        st.image(image, caption=source_caption, use_container_width=True)
    with right:
        format_result = validate_format(result['number'], format_name)
        if result['accepted'] and format_result['valid']:
            st.success('Recognized number')
            st.code(result['number'], language=None)
            st.caption(format_result['message'])
        elif result['accepted']:
            st.warning('Number recognized, but it does not match the selected format.')
            st.code(result['number'], language=None)
            st.write(format_result['message'])
        else:
            st.warning('Input rejected — please review or rewrite it.')
            st.write(result['reason'])
            if result.get('candidate_number'):
                st.caption('Digit CNN forced guess — not accepted')
                st.code(result['candidate_number'], language=None)
        st.markdown('#### Dual-path verification')
        if result.get('sequence_available'):
            first, second = st.columns(2)
            with first:
                st.metric('Component pathway', result.get('candidate_number') or 'No reading')
                st.caption('Digit CNN + validator + digit/letter gate')
            with second:
                st.metric('Full-number CRNN', result.get('sequence_reading') or 'No reading')
                st.caption(f"CNN + BiLSTM + CTC • confidence {result['sequence_confidence']:.2f}")
            component = result.get('candidate_number')
            sequence = result.get('sequence_reading')
            if result['accepted']:
                st.success('Agreement confirmed: both recognition pathways returned the same safe number.')
            elif component and sequence and component != sequence:
                st.error('Rejected by dual-path verification: the two pathways disagree.')
            elif component == sequence:
                st.warning('Both pathways produced the same candidate, but an independent safety check rejected it.')
            else:
                st.warning('Rejected before agreement: at least one pathway could not produce a safe reading.')
        else:
            st.info('Full-number CRNN is not trained yet. Train it to enable dual-path agreement.')
        st.caption(f'Processing time: {elapsed:.0f} ms (including first-call overhead when applicable)')

    marked = Image.fromarray(result['ink']).convert('RGB')
    draw = ImageDraw.Draw(marked)
    for i, box in enumerate(result['boxes']):
        draw.rectangle(box, outline='#00cc99', width=2)
        draw.text((box[0], max(0, box[1]-12)), str(i+1), fill='#ff9933')
    with st.expander('Inspect how the system made its decision', expanded=True):
        st.image(marked, caption='Candidate components, ordered from left to right')
        if result['crops']:
            previews = [Image.fromarray((x[..., 0]*255).astype('uint8')).resize((112, 112))
                        for x in result['crops']]
            st.image(previews, caption=[f'Component {i+1}' for i in range(len(previews))])
        if result['components']:
            st.dataframe(result['components'], use_container_width=True)
        st.write('Acceptance thresholds:', {k: recognizer.thresholds[k]
                                             for k in ('classifier', 'validator', 'character')})
        st.caption('Scores are model outputs, not guaranteed probabilities of correctness. Every component must pass all three checks; once trained, the full-number CRNN must agree too.')
    export = {k: v for k, v in result.items() if k not in ('ink', 'crops')}
    st.download_button('Download prediction details', json.dumps(export, indent=2),
                       'prediction.json', 'application/json')
    with st.expander('Correct this result and save private feedback'):
        st.caption('Feedback stays in this project folder. It is not sent anywhere automatically.')
        actual = st.text_input('What did you actually write? Use digits only, or write NON_DIGIT.',
                               key=f'feedback-{source_caption}')
        if st.button('Save feedback sample', key=f'save-{source_caption}'):
            if not actual.strip():
                st.warning('Enter the actual label before saving.')
            elif actual.strip() != 'NON_DIGIT' and not actual.strip().isdigit():
                st.warning('Use digits only, or exactly NON_DIGIT.')
            else:
                save_feedback(image, result, actual, source_caption, format_name)
                st.success('Saved locally in feedback/feedback.csv.')


try:
    recognizer = get_recognizer()
except FileNotFoundError as exc:
    st.info('The interface is ready. Train the models before recognizing handwriting.')
    st.code('python -m hdr.prepare\npython -m hdr.train\npython -m hdr.prepare_character --emnist-zip "C:\\\\path\\\\to\\\\gzip.zip"\npython -m hdr.train_character\npython -m hdr.train_sequence\npython -m hdr.evaluate')
    st.caption(str(exc))
    st.stop()

training_path = ROOT/'models'/'training.json'
if training_path.exists() and json.loads(training_path.read_text()).get('smoke_training'):
    st.warning('These are smoke-test models. Retrain on the full dataset before reporting accuracy.')

format_name = st.selectbox('Expected input type', list(FORMATS), index=0)
st.caption(FORMATS[format_name][0])

draw_tab, upload_tab, camera_tab, dashboard_tab = st.tabs(
    ['Draw a number', 'Upload an image', 'Live camera', 'Analysis dashboard'])

with draw_tab:
    st.write('Use black ink, keep digits separate, and write one horizontal row.')
    pen_width = st.slider('Pen width', min_value=6, max_value=30, value=16, step=2)
    canvas = st_canvas(
        fill_color='rgba(255, 255, 255, 0)',
        stroke_width=pen_width,
        stroke_color='#000000',
        background_color='#FFFFFF',
        height=280,
        width=800,
        drawing_mode='freedraw',
        display_toolbar=True,
        update_streamlit=True,
        key='digit-canvas',
    )
    if st.button('Evaluate drawing', type='primary'):
        if canvas.image_data is None:
            st.warning('Draw a number before evaluating it.')
        else:
            drawing = Image.fromarray(np.asarray(canvas.image_data, dtype=np.uint8), mode='RGBA')
            show_result(drawing, 'Canvas drawing', format_name)

with upload_tab:
    file = st.file_uploader('Choose a handwriting image', type=['png', 'jpg', 'jpeg', 'bmp', 'webp'])
    if file:
        try:
            with Image.open(file) as opened:
                if opened.width*opened.height > 16_000_000:
                    st.error('Use an image smaller than 16 megapixels.')
                    st.stop()
                uploaded = opened.copy()
            show_result(uploaded, 'Uploaded input', format_name)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            st.error(f'Could not process this image: {exc}')

with camera_tab:
    st.write('Start the camera, hold one horizontal handwritten number inside the frame, then capture five frames.')
    st.caption('The app accepts a live-camera reading only when at least three captured frames produce the same safe result.')
    if not WEBRTC_AVAILABLE:
        st.error('Live camera support is not installed. Run: python -m pip install -r requirements.txt')
    else:
        context = webrtc_streamer(key='digitlab-live-camera', media_stream_constraints={
            'video': True, 'audio': False})
        if st.button('Evaluate five live frames', key='evaluate-live'):
            if not context.state.playing or context.video_receiver is None:
                st.warning('Click START and allow camera access first.')
            else:
                readings, captured, accepted_frames = [], None, []
                for _ in range(5):
                    try:
                        frame = context.video_receiver.get_frame(timeout=2)
                        captured = Image.fromarray(frame.to_ndarray(format='rgb24'))
                        item = recognizer.predict(captured)
                        if item['accepted']:
                            readings.append(item['number'])
                            accepted_frames.append((item['number'], captured.copy()))
                    except Exception:
                        continue
                if captured is None:
                    st.warning('No camera frames were received. Keep the camera running and try again.')
                elif readings:
                    winner = max(set(readings), key=readings.count)
                    st.caption(f'Consistent safe readings: {readings.count(winner)} of 5')
                    if readings.count(winner) >= 3:
                        winner_image = next(image for value, image in accepted_frames if value == winner)
                        show_result(winner_image, 'Live camera frame', format_name)
                    else:
                        st.warning('The camera readings were unstable. Hold the paper steady, improve lighting, and try again.')
                else:
                    st.warning('None of the live frames passed the recognition safety checks.')

with dashboard_tab:
    st.subheader('Selective OCR system dashboard')
    st.write('The final answer is shown only after independent component checks and, once trained, full-number sequence agreement.')
    report_path = ROOT/'reports'/'evaluation.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        columns = st.columns(4)
        columns[0].metric('Digit accuracy', f"{report['digit_accuracy']:.2%}")
        columns[1].metric('Negative false acceptance', f"{report['pipeline_negative_false_accept_rate']:.2%}")
        columns[2].metric('String exact accuracy', f"{report['synthetic_strings']['exact_string_accuracy']:.2%}")
        columns[3].metric('String coverage', f"{report['synthetic_strings']['coverage']:.2%}")
        st.json(report['thresholds'])
    else:
        st.info('Run python -m hdr.evaluate to populate the evaluation dashboard.')
    feedback_csv = ROOT/'feedback'/'feedback.csv'
    if feedback_csv.exists():
        with feedback_csv.open(encoding='utf-8') as handle:
            feedback = list(csv.DictReader(handle))
        st.metric('Saved feedback samples', len(feedback))
        st.dataframe(feedback, use_container_width=True)
    else:
        st.caption('No feedback samples saved yet.')

with st.expander('How the recognition pipeline works'):
    st.write('1. Preprocessing locates the writing and converts each candidate to a centered 28 × 28 image.')
    st.write('2. The digit CNN predicts 0–9. The validator separately estimates whether the crop resembles a digit.')
    st.write('3. A 36-class character CNN checks whether the crop looks more like a digit or a letter.')
    st.write('4. A CNN–BiLSTM–CTC sequence model reads the complete number after training. It must agree with the component pathway.')
    st.write('5. All checks must pass validation-set thresholds. Accepted digits are joined as text, preserving leading zeros.')
    st.write('A letter such as O can resemble 0. This system cannot guarantee rejection of every non-digit.')
