import React, {useEffect, useRef, useState} from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

const formats = ['General number', 'Indian mobile number', 'Indian PIN code', 'OTP (6 digits)', 'Student/roll number']
const formatHelp = {
  'General number': 'Any 1–12 digit number.',
  'Indian mobile number': '10 digits, beginning with 6, 7, 8 or 9.',
  'Indian PIN code': '6 digits, beginning with 1–8.',
  'OTP (6 digits)': 'Exactly 6 digits.',
  'Student/roll number': 'Exactly 8 digits.',
}

function App() {
  const [tab, setTab] = useState('draw')
  const [format, setFormat] = useState('General number')
  const [result, setResult] = useState(null)
  const [preview, setPreview] = useState(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [health, setHealth] = useState(null)
  const [dashboard, setDashboard] = useState(null)
  const [actual, setActual] = useState('')
  const [sourceImage, setSourceImage] = useState(null)
  const [feedbackSaving, setFeedbackSaving] = useState(false)
  const canvasRef = useRef(null); const drawing = useRef(false); const videoRef = useRef(null); const streamRef = useRef(null)

  useEffect(() => { fetch('/api/health').then(r => r.json()).then(setHealth).catch(() => setMessage('Backend is offline. Start python server.py first.')) }, [])
  useEffect(() => { if (tab === 'insights') fetch('/api/dashboard').then(r => r.json()).then(setDashboard).catch(() => {}) }, [tab])
  useEffect(() => () => streamRef.current?.getTracks().forEach(t => t.stop()), [])

  const sendImage = async (blob, source='web') => {
    setBusy(true); setMessage(''); setActual('')
    const form = new FormData(); form.append('file', blob, 'handwriting.png'); form.append('format_name', format)
    try {
      const response = await fetch('/api/predict', {method:'POST', body:form}); const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Could not evaluate this image.')
      setResult(data)
    } catch (error) { setMessage(error.message) } finally { setBusy(false) }
  }
  const canvasPoint = (event) => {
    const canvas = canvasRef.current; const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width, scaleY = canvas.height / rect.height
    return [(event.clientX-rect.left)*scaleX, (event.clientY-rect.top)*scaleY]
  }
  const drawStart = (e) => { drawing.current = true; const [x,y] = canvasPoint(e); const ctx=canvasRef.current.getContext('2d'); ctx.beginPath(); ctx.moveTo(x,y) }
  const drawMove = (e) => { if(!drawing.current) return; const [x,y]=canvasPoint(e); const ctx=canvasRef.current.getContext('2d'); ctx.lineTo(x,y); ctx.stroke() }
  const setupCanvas = (node) => { if (!node || canvasRef.current) return; canvasRef.current=node; const ctx=node.getContext('2d'); ctx.fillStyle='#fffdf9'; ctx.fillRect(0,0,node.width,node.height); ctx.strokeStyle='#28211e'; ctx.lineWidth=16; ctx.lineCap='round'; ctx.lineJoin='round' }
  const clearCanvas = () => { const c=canvasRef.current; const ctx=c.getContext('2d'); ctx.fillStyle='#fffdf9';ctx.fillRect(0,0,c.width,c.height); setPreview(null);setSourceImage(null);setResult(null) }
  const evaluateDrawing = () => canvasRef.current.toBlob(blob => { if(!blob) return setMessage('Could not capture the drawing. Please try again.'); setPreview(URL.createObjectURL(blob)); setSourceImage(blob); sendImage(blob,'drawing') }, 'image/png')
  const upload = (e) => { const file=e.target.files?.[0]; if(file){setPreview(URL.createObjectURL(file));setSourceImage(file);sendImage(file,'upload')} }
  const cameraStart = async () => { try { streamRef.current=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'},audio:false});videoRef.current.srcObject=streamRef.current } catch { setMessage('Camera access was not granted. Allow it in your browser and try again.') } }
  const cameraEvaluate = async () => {
    if(!videoRef.current?.videoWidth) return setMessage('Start the camera and wait for its preview first.')
    setBusy(true);setMessage('Checking five frames for a stable reading…');const readings=[]
    let errors=0
    for(let i=0;i<5;i++) {
      const c=document.createElement('canvas'); const scale=Math.min(1,1280/videoRef.current.videoWidth)
      c.width=Math.round(videoRef.current.videoWidth*scale); c.height=Math.round(videoRef.current.videoHeight*scale)
      c.getContext('2d').drawImage(videoRef.current,0,0,c.width,c.height)
      const blob=await new Promise(r=>c.toBlob(r,'image/jpeg',.9)); const form=new FormData();form.append('file',blob,'camera.jpg');form.append('format_name',format);form.append('source','camera')
      try { const r=await fetch('/api/predict',{method:'POST',body:form}); const d=await r.json(); if(!r.ok) throw new Error(d.detail||'Scan failed'); if(d.candidate_reading||d.sequence_reading) readings.push({result:d,blob}) } catch { errors++ }
      await new Promise(r=>setTimeout(r,320))
    }
    if(!readings.length){setMessage(errors===5?'Camera frames could not reach the recognition service. Check that python server.py is still running.':'No digits were detected. Keep the number centred, use good light, and move closer.');setBusy(false);return}
    const key=x=>x.result.reading||x.result.candidate_reading||x.result.sequence_reading
    const winner=readings.sort((a,b)=>readings.filter(x=>key(x)===key(b)).length-readings.filter(x=>key(x)===key(a)).length)[0]
    const matches=readings.filter(x=>key(x)===key(winner)).length
    setPreview(URL.createObjectURL(winner.blob));setSourceImage(winner.blob);setResult(winner.result)
    setMessage(matches>=3?`Camera scanned ${key(winner)} consistently in ${matches} of 5 frames.`:`Camera readings varied; displaying its best scan: ${key(winner)}. Hold still and try again for a stable result.`);setBusy(false)
  }
  const saveFeedback = async () => {
    if(!actual || (!/^\d+$/.test(actual) && actual!=='NON_DIGIT')) return setMessage('Enter the true value using digits only, or exactly NON_DIGIT.')
    if(!sourceImage) return setMessage('The original image is unavailable. Evaluate the input once more, then save feedback.')
    setFeedbackSaving(true); setMessage('Saving labelled feedback…')
    try {
      const f=new FormData();f.append('file',sourceImage,'feedback.png');f.append('actual_label',actual);f.append('format_name',format);f.append('source',tab)
      const r=await fetch('/api/feedback',{method:'POST',body:f}); const d=await r.json()
      if(!r.ok || !d.saved) throw new Error(d.detail||'Feedback could not be saved.')
      setActual(''); setMessage('Feedback saved. The labelled image and its result are now in the feedback folder.')
      fetch('/api/dashboard').then(x=>x.json()).then(setDashboard).catch(()=>{})
    } catch(error) { setMessage(error.message||'Feedback could not be saved.') } finally { setFeedbackSaving(false) }
  }

  return <main>
    <header className="hero"><div><div className="eyebrow">SELECTIVE HANDWRITING OCR</div><h1>DigitLab <i>Studio</i></h1><p>Read handwritten numbers with two independent recognition paths—and know when not to trust an answer.</p></div><div className="hero-right"><div className="status-dot"></div><span>{health?.sequence_ready ? 'Dual-path system online' : 'Safety checks online'}</span><div className="hero-stat"><b>3</b><small>safety gates</small><b>2</b><small>reading paths</small></div></div></header>
    <section className="control-bar"><div><label>EXPECTED INPUT TYPE</label><select value={format} onChange={e=>setFormat(e.target.value)}>{formats.map(x=><option key={x}>{x}</option>)}</select><p>{formatHelp[format]}</p></div><div className="trust"><span>◒ Private by default</span><span>◈ Explainable decision</span><span>⌁ Human feedback loop</span></div></section>
    <nav className="tabs">{[['draw','✦ Draw'],['upload','↥ Upload'],['camera','◉ Live camera'],['insights','▦ Insights']].map(([id,label])=><button key={id} className={tab===id?'active':''} onClick={()=>setTab(id)}>{label}</button>)}</nav>
    {message && <div className="notice">{message}</div>}
    {tab==='draw' && <section className="workspace"><div className="panel ink-panel"><div className="panel-head"><div><span className="kicker">01 / INPUT</span><h2>Write naturally.</h2><p>Use one horizontal row. Keep digits separated and use a dark, steady stroke.</p></div><button className="ghost" onClick={clearCanvas}>Clear canvas</button></div><div className="canvas-shell"><canvas ref={setupCanvas} width="1000" height="360" onPointerDown={drawStart} onPointerMove={drawMove} onPointerUp={()=>drawing.current=false} onPointerLeave={()=>drawing.current=false}/></div><button className="primary" disabled={busy} onClick={evaluateDrawing}>{busy?'Analysing…':'Evaluate drawing →'}</button></div><Guide/></section>}
    {tab==='upload' && <section className="workspace"><div className="panel upload-panel"><span className="kicker">01 / INPUT</span><h2>Upload handwriting.</h2><p>PNG, JPG, JPEG, BMP or WEBP · one clean row · under 12 MB</p><label className="dropzone"><input type="file" accept="image/png,image/jpeg,image/bmp,image/webp" onChange={upload}/><b>Choose an image</b><span>or drag a file here</span></label></div><Guide/></section>}
    {tab==='camera' && <section className="workspace"><div className="panel camera-panel"><span className="kicker">LIVE / STABILITY CHECK</span><h2>Read it in the moment.</h2><p>Place one handwritten row inside the centre guide. The camera scan removes uneven lighting and ignores most background.</p><div className="video-shell"><video ref={videoRef} autoPlay playsInline muted/><div className="scan-guide"><span>KEEP DIGITS IN THIS BAND</span></div></div><div className="action-row"><button className="ghost" onClick={cameraStart}>Start camera</button><button className="primary" disabled={busy} onClick={cameraEvaluate}>{busy?'Checking frames…':'Evaluate five frames →'}</button></div></div><Guide/></section>}
    {tab==='insights' && <Insights dashboard={dashboard}/>} {result && <Result result={result} preview={preview} actual={actual} setActual={setActual} saveFeedback={saveFeedback} feedbackSaving={feedbackSaving}/>} 
    <footer>DigitLab Studio · Local research prototype · The system rejects uncertain handwriting rather than inventing certainty.</footer>
  </main>
}

function Guide(){return <aside className="guide"><span className="kicker">HOW IT DECIDES</span><div className="guide-step"><b>01</b><p><strong>Component pathway</strong><br/>Digit CNN, digit validator and digit-vs-letter gate inspect every character.</p></div><div className="guide-step"><b>02</b><p><strong>Sequence pathway</strong><br/>CNN + BiLSTM + CTC reads the complete number as one string.</p></div><div className="guide-step"><b>03</b><p><strong>Selective decision</strong><br/>A result is accepted only when safety checks pass and both paths agree.</p></div></aside>}
function Result({result,preview,actual,setActual,saveFeedback,feedbackSaving}){const accepted=result.decision==='accepted';const mismatch=result.sequence_available&&result.candidate_reading&&result.sequence_reading!==result.candidate_reading;const recovered=result.decision_mode==='sequence_supported';return <section className="result-wrap"><div className="result-image">{preview&&<img src={preview}/>}<span>Input under review</span></div><div className="result-main"><span className={'decision '+result.decision}>{accepted?(recovered?'SEQUENCE-CONFIRMED READING':'SAFE READING'):result.decision==='format_warning'?'FORMAT REVIEW':'REVIEW REQUIRED'}</span><h2>{accepted?result.reading:result.candidate_reading||'No safe reading'}</h2><p>{accepted?(recovered?'The full-number model and digit/letter checks confirmed this handwriting. A strict component-validator score was relaxed only because both paths agreed.':'All independent checks agreed on this reading.'):result.reason||result.format?.message}</p><div className="path-grid"><div><small>COMPONENT PATHWAY</small><b>{result.candidate_reading||'—'}</b><span>Digit CNN + 2 gates</span></div><div><small>FULL-NUMBER CRNN</small><b>{result.sequence_available?(result.sequence_reading||'—'):'Not trained'}</b><span>{result.sequence_available?`Frame confidence ${Number(result.sequence_confidence).toFixed(2)}`:'CNN + BiLSTM + CTC'}</span></div></div>{accepted?<div className="agreement">✓ {recovered?'Sequence-supported agreement confirmed.':'Strict agreement confirmed: both paths returned the same safe number.'}</div>:mismatch?<div className="rejection">× Rejected by dual-path verification: the two paths disagree.</div>:<div className="rejection">× Rejected by an independent safety or format check.</div>}<details><summary>Inspect component scores & thresholds</summary><pre>{JSON.stringify({components:result.components,thresholds:result.thresholds},null,2)}</pre></details><div className="feedback"><div><b>Help it improve</b><span>Enter what was actually written, then save this labelled sample locally.</span></div><input value={actual} onChange={e=>setActual(e.target.value.toUpperCase())} placeholder="Actual digits or NON_DIGIT"/><button className="ghost" disabled={feedbackSaving} onClick={saveFeedback}>{feedbackSaving?'Saving…':'Save feedback'}</button></div></div></section>}
function Insights({dashboard}){const r=dashboard?.report;return <section className="insights"><div className="panel insight-top"><span className="kicker">MODEL HEALTH</span><h2>Evidence, not just a prediction.</h2><p>Metrics below are measured after training; feedback samples remain local to this project.</p><div className="metric-grid"><Metric label="Digit accuracy" value={r?`${(r.digit_accuracy*100).toFixed(2)}%`:'—'}/><Metric label="Negative false acceptance" value={r?`${(r.pipeline_negative_false_accept_rate*100).toFixed(2)}%`:'—'}/><Metric label="String exact accuracy" value={r?`${(r.synthetic_strings.exact_string_accuracy*100).toFixed(1)}%`:'—'}/><Metric label="Feedback samples" value={dashboard?.feedback_count??'—'}/></div></div><div className="panel"><span className="kicker">DECISION POLICY</span><h3>Acceptance is selective.</h3><p>We report coverage and rejection instead of hiding uncertain cases behind a forced answer. This is why the application can refuse a result even when one model has guessed a digit.</p></div></section>}
function Metric({label,value}){return <div className="metric"><b>{value}</b><span>{label}</span></div>}
createRoot(document.getElementById('root')).render(<App />)
