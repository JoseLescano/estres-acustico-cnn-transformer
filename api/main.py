from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from src.config import (
    AUDIO_DURATION_SECONDS,
    DEVICE,
    GABOR_SIGMA,
    HOP_LENGTH,
    MODELS_SAVED_DIR,
    N_FFT,
    SAMPLE_RATE,
    SOUND_LEVEL_REFERENCE_DB,
)
from src.data.gabor_spectrogram import (
    compute_gabor_spectrogram,
    normalize_spectrogram,
    pad_or_trim_spectrogram,
)
from src.data.load_dataset import AcousticStressDataset
from src.models.hybrid_model import HybridCNNTransformer

app = FastAPI(title="Acoustic Stress Dashboard API")
_model: HybridCNNTransformer | None = None


def _load_model() -> HybridCNNTransformer:
    global _model
    if _model is not None:
        return _model

    checkpoint_path = MODELS_SAVED_DIR / "best_model.pt"
    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=503,
            detail="No hay un modelo entrenado. Ejecuta primero src.training.train.",
        )

    model = HybridCNNTransformer().to(DEVICE)
    target_frames = AcousticStressDataset._compute_expected_frames()
    dummy = torch.zeros(1, 1, N_FFT // 2 + 1, target_frames, device=DEVICE)
    with torch.no_grad():
        model(dummy)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _model = model
    return model


def _audio_to_tensor(audio_bytes: bytes) -> tuple[torch.Tensor, float, float]:
    try:
        audio, sample_rate = sf.read(
            BytesIO(audio_bytes), dtype="float32", always_2d=False
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail="No se pudo leer el WAV.") from exc

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        raise HTTPException(
            status_code=422,
            detail=f"El audio debe tener {SAMPLE_RATE} Hz; recibió {sample_rate} Hz.",
        )

    target_length = int(SAMPLE_RATE * AUDIO_DURATION_SECONDS)
    audio = audio[:target_length]
    if audio.size < target_length:
        audio = np.pad(audio, (0, target_length - audio.size))

    peak_amplitude = float(np.max(np.abs(audio)))
    rms_amplitude = float(np.sqrt(np.mean(np.square(audio))))
    lafmax = SOUND_LEVEL_REFERENCE_DB + 20.0 * np.log10(max(peak_amplitude, 1e-12))
    laeq = SOUND_LEVEL_REFERENCE_DB + 20.0 * np.log10(max(rms_amplitude, 1e-12))

    spectrogram = compute_gabor_spectrogram(
        audio,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        sigma=GABOR_SIGMA,
    )
    spectrogram = normalize_spectrogram(spectrogram)
    spectrogram = pad_or_trim_spectrogram(
        spectrogram, AcousticStressDataset._compute_expected_frames()
    )
    tensor = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    return tensor, max(0.0, laeq), max(0.0, lafmax)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_available": (MODELS_SAVED_DIR / "best_model.pt").exists(),
    }


@app.get("/", response_class=HTMLResponse)
def dashboard_index() -> str:
    dashboard_path = Path(__file__).with_name("dashboard.html")
    return dashboard_path.read_text(encoding="utf-8")

    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Acoustic Signal Lab</title>
      <style>
        :root {
          --paper: #f3efe6;
          --paper-strong: #e9e3d6;
          --ink: #18252c;
          --muted: #5d686d;
          --line: rgba(24, 37, 44, 0.5);
          --accent: #f08a5d;
          --accent-soft: rgba(240, 138, 93, 0.18);
          --mint: #bfead7;
          --mint-strong: #9cd7c2;
          --shadow: rgba(24, 37, 44, 0.12);
          --danger: #d96842;
          --warning: #d1a85b;
          --success: #4b8b73;
        }

        * { box-sizing: border-box; }

        body {
          margin: 0;
          background: var(--paper);
          color: var(--ink);
          font-family: Georgia, 'Times New Roman', serif;
        }

        main {
          max-width: 1360px;
          margin: 0 auto;
          padding: 28px 32px 48px;
        }

        .topbar {
          display: flex;
          align-items: flex-end;
          justify-content: space-between;
          gap: 24px;
          border-bottom: 2px solid var(--line);
          padding-bottom: 18px;
        }

        .eyebrow {
          font-family: Arial, Helvetica, sans-serif;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          font-size: 12px;
          font-weight: 700;
          color: var(--accent);
          margin-bottom: 8px;
        }

        h1 {
          font-size: clamp(4rem, 7vw, 9rem);
          line-height: 0.9;
          font-weight: 400;
          letter-spacing: -0.06em;
          margin: 0;
          max-width: 780px;
        }

        .status-pill {
          border: 1px solid rgba(24, 37, 44, 0.3);
          background: var(--mint);
          color: var(--ink);
          font: 700 14px/1 Arial, Helvetica, sans-serif;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          padding: 12px 18px;
          border-radius: 10px;
          box-shadow: 0 10px 22px rgba(24, 37, 44, 0.05);
          min-width: 160px;
          text-align: center;
          transition: all 0.2s ease;
        }

        .status-pill.offline {
          background: #f1d6cd;
          color: #7d3c2d;
        }

        .grid {
          display: grid;
          grid-template-columns: 1.2fr 0.8fr;
          gap: 28px;
          margin-top: 30px;
        }

        .panel {
          border-top: 3px solid var(--ink);
          padding-top: 16px;
        }

        .panel h2 {
          margin: 0 0 18px;
          font-size: clamp(1.8rem, 1.7vw, 2.5rem);
          font-weight: 400;
          letter-spacing: -0.04em;
        }

        .metric-box {
          background: rgba(255,255,255,0.04);
          border-radius: 16px;
          padding: 10px 0 14px;
        }

        .metric-value {
          font-size: clamp(4.5rem, 6vw, 8rem);
          line-height: 0.9;
          letter-spacing: -0.08em;
          margin: 0;
        }

        .inline-status {
          display: inline-flex;
          align-items: center;
          gap: 10px;
          margin-top: 12px;
          font: 16px/1.4 Arial, Helvetica, sans-serif;
          color: var(--muted);
        }

        .dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          background: var(--success);
          box-shadow: 0 0 0 6px rgba(75, 139, 115, 0.12);
          display: inline-block;
        }

        .dot.alert { background: var(--warning); box-shadow: 0 0 0 6px rgba(209, 168, 91, 0.12); }
        .dot.danger { background: var(--danger); box-shadow: 0 0 0 6px rgba(217, 104, 66, 0.14); }

        .chart-shell {
          position: relative;
          margin-top: 18px;
          background: rgba(120, 112, 96, 0.065);
          border: 1px solid rgba(24, 37, 44, 0.1);
          border-radius: 12px;
          overflow: hidden;
          min-height: 260px;
        }

        .chart-tooltip {
          position: absolute;
          left: 0;
          top: 0;
          transform: translate(-50%, -110%);
          background: rgba(24, 37, 44, 0.96);
          color: #f8f6f1;
          padding: 7px 10px;
          border-radius: 8px;
          font: 700 12px/1 Arial, Helvetica, sans-serif;
          letter-spacing: 0.04em;
          pointer-events: none;
          opacity: 0;
          transition: opacity 0.12s ease;
          white-space: nowrap;
          box-shadow: 0 10px 18px rgba(24, 37, 44, 0.18);
          border: 1px solid rgba(255,255,255,0.08);
        }

        canvas {
          display: block;
          width: 100%;
          height: 260px;
          background: linear-gradient(180deg, rgba(255,255,255,0.08), transparent);
        }

        .card {
          background: rgba(255,255,255,0.06);
          border: 1px solid rgba(24, 37, 44, 0.18);
          border-radius: 16px;
          padding: 18px 18px 16px;
          box-shadow: 0 12px 24px var(--shadow);
        }

        .upload-wrap {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .upload-label {
          display: block;
          border: 1px dashed rgba(24, 37, 44, 0.42);
          border-radius: 12px;
          padding: 18px 16px;
          background: rgba(255,255,255,0.04);
          color: var(--muted);
          cursor: pointer;
          transition: border-color 0.2s ease, transform 0.2s ease, background 0.2s ease;
          font: 16px Arial, Helvetica, sans-serif;
        }

        .upload-label:hover {
          border-color: rgba(24, 37, 44, 0.7);
          background: rgba(255,255,255,0.12);
          transform: translateY(-1px);
        }

        .upload-label input { display: none; }

        .file-meta {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          align-items: center;
          color: var(--muted);
          font: 13px Arial, Helvetica, sans-serif;
          min-height: 22px;
        }

        .button {
          appearance: none;
          border: none;
          background: linear-gradient(180deg, rgba(24, 37, 44, 0.96), rgba(12, 22, 28, 0.98));
          color: #f8f6f1;
          padding: 16px 22px;
          border-radius: 12px;
          font: 700 18px/1 Arial, Helvetica, sans-serif;
          cursor: pointer;
          transition: transform 0.2s ease, box-shadow 0.2s ease, opacity 0.2s ease;
          box-shadow: 0 12px 22px rgba(24, 37, 44, 0.18);
        }

        .button:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 18px 28px rgba(24, 37, 44, 0.2);
        }

        .button:disabled {
          opacity: 0.7;
          cursor: wait;
        }

        .result-panel {
          margin-top: 20px;
          background: linear-gradient(180deg, rgba(255,255,255,0.12), rgba(255,255,255,0.04));
          border: 1px solid rgba(24, 37, 44, 0.14);
          border-radius: 12px;
          padding: 16px 18px;
          min-height: 100px;
          display: none;
        }

        .result-panel.visible { display: block; }

        .result-panel strong {
          font-size: 0.82rem;
          text-transform: uppercase;
          letter-spacing: 0.11em;
          color: var(--muted);
          font-family: Arial, Helvetica, sans-serif;
        }

        .result-value {
          font-size: clamp(2.2rem, 3vw, 3.6rem);
          line-height: 1;
          margin: 8px 0 6px;
          letter-spacing: -0.06em;
        }

        .result-meta {
          font: 14px/1.4 Arial, Helvetica, sans-serif;
          color: var(--muted);
        }

        .hint {
          margin-top: 16px;
          font: 16px/1.55 Arial, Helvetica, sans-serif;
          color: var(--muted);
        }

        .legend {
          margin-top: 34px;
          border-top: 3px solid var(--ink);
          padding-top: 16px;
        }

        .legend-header {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 18px;
          margin-bottom: 16px;
        }

        .legend h2 {
          margin: 0;
          font-size: clamp(1.8rem, 1.7vw, 2.5rem);
          font-weight: 400;
          letter-spacing: -0.04em;
        }

        .legend-note {
          margin: 0;
          color: var(--muted);
          font: 13px/1.4 Arial, Helvetica, sans-serif;
          text-align: right;
        }

        .legend-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 10px;
        }

        .legend-item {
          display: grid;
          grid-template-columns: 12px 1fr;
          gap: 12px;
          min-height: 92px;
          padding: 14px;
          background: rgba(255,255,255,0.08);
          border: 1px solid rgba(24, 37, 44, 0.16);
          border-radius: 10px;
        }

        .legend-swatch {
          width: 12px;
          height: 100%;
          min-height: 62px;
          border-radius: 5px;
        }

        .legend-item.low .legend-swatch { background: #62f862; }
        .legend-item.medium .legend-swatch { background: #f5f957; }
        .legend-item.high .legend-swatch { background: #ff5054; }
        .legend-item.very-high .legend-swatch { background: #999999; }

        .legend-title {
          display: block;
          margin-bottom: 6px;
          font: 700 15px/1.25 Arial, Helvetica, sans-serif;
          color: var(--ink);
        }

        .legend-range {
          display: block;
          color: var(--muted);
          font: 14px/1.35 Arial, Helvetica, sans-serif;
        }

        @media (max-width: 900px) {
          .grid {
            grid-template-columns: 1fr;
          }

          .legend-grid {
            grid-template-columns: repeat(2, 1fr);
          }

          .topbar {
            align-items: flex-start;
            flex-direction: column;
          }

          .status-pill {
            width: 100%;
          }
        }

        @media (max-width: 560px) {
          main { padding-left: 18px; padding-right: 18px; }

          .legend-header {
            display: block;
          }

          .legend-note {
            margin-top: 6px;
            text-align: left;
          }

          .legend-grid {
            grid-template-columns: 1fr;
          }
        }
      </style>
    </head>
    <body>
      <main>
        <header class="topbar">
          <div>
            <div class="eyebrow">Acoustic Signal Lab / 01</div>
            <h1>Mapa de estrés acústico</h1>
          </div>
          <div class="status-pill" id="health">Comprobando API...</div>
        </header>

        <div class="grid">
          <section class="panel">
            <h2>Lectura reciente</h2>
            <div class="metric-box">
              <div class="metric-value" id="metric">--</div>
              <div class="inline-status">
                <span class="dot" id="statusDot"></span>
                <span id="state">Cargando historial...</span>
              </div>
            </div>
            <div class="chart-shell">
              <canvas id="chart" width="700" height="260"></canvas>
              <div id="chartTooltip" class="chart-tooltip">0.000</div>
            </div>
          </section>

          <section class="panel">
            <h2>Analizar archivo WAV</h2>
            <div class="card upload-wrap">
              <label class="upload-label" for="audio">
                <input id="audio" type="file" accept=".wav,audio/wav" />
                <span id="file-label">Seleccionar archivo</span>
              </label>
              <div class="file-meta">
                <span id="file-name">Sin archivos seleccionados</span>
              </div>

              <button class="button" id="submit" type="button">Ejecutar predicción</button>

              <div class="result-panel" id="resultPanel">
                <strong>Resultado</strong>
                <div class="result-value" id="resultValue">--</div>
                <div class="result-meta" id="resultMeta">Esperando análisis...</div>
              </div>
            </div>

            <p class="hint">El modelo recibe audio mono a 22.050 Hz y analiza una ventana de 4 segundos para estimar el índice de estrés acústico.</p>
          </section>
        </div>

        <section class="legend" aria-labelledby="legend-title">
          <div class="legend-header">
            <h2 id="legend-title">Referencia de niveles acústicos</h2>
            <p class="legend-note">Monitoreo de ruido no estacionario · distrito de Ate</p>
          </div>
          <div class="legend-grid">
            <div class="legend-item low">
              <span class="legend-swatch" aria-hidden="true"></span>
              <div>
                <strong class="legend-title">Nivel bajo · verde</strong>
                <span class="legend-range">Rango: 70 a 75 dB</span>
              </div>
            </div>
            <div class="legend-item medium">
              <span class="legend-swatch" aria-hidden="true"></span>
              <div>
                <strong class="legend-title">Nivel medio · amarillo</strong>
                <span class="legend-range">Rango: 75 a 80 dB</span>
              </div>
            </div>
            <div class="legend-item high">
              <span class="legend-swatch" aria-hidden="true"></span>
              <div>
                <strong class="legend-title">Nivel alto · rojo</strong>
                <span class="legend-range">Rango: 80 a 85 dB</span>
              </div>
            </div>
            <div class="legend-item very-high">
              <span class="legend-swatch" aria-hidden="true"></span>
              <div>
                <strong class="legend-title">Nivel muy alto · plomo</strong>
                <span class="legend-range">Rango: 85 a 90 dB</span>
              </div>
            </div>
          </div>
        </section>
      </main>

      <script>
        const $ = (id) => document.getElementById(id);

        function classifyStress(value) {
          if (value >= 0.75) return { label: 'Alto', className: 'danger' };
          if (value >= 0.45) return { label: 'Medio', className: 'alert' };
          return { label: 'Normal', className: 'success' };
        }

        function setStatusIndicator(state, value) {
          const dot = $('statusDot');
          const stress = classifyStress(Number(value) || 0);
          dot.className = 'dot ' + stress.className;
          if (state) {
            $('state').textContent = `Estado: ${state} / ventana de observación`;
          }
        }

        let dashboardHistory = [0.35, 0.41, 0.42];
        let animationFrameId = null;

        function draw(values, animate = false) {
          const canvas = $('chart');
          const ctx = canvas.getContext('2d');
          const w = canvas.width;
          const h = canvas.height;

          ctx.clearRect(0, 0, w, h);
          ctx.fillStyle = '#f3efe6';
          ctx.fillRect(0, 0, w, h);

          ctx.strokeStyle = 'rgba(24,37,44,0.12)';
          ctx.lineWidth = 1;
          for (let i = 0; i <= 4; i++) {
            const y = (h / 4) * i;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
          }

          for (let i = 0; i <= 6; i++) {
            const x = (w / 6) * i;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
          }

          if (!values || values.length < 2) return;

          const maxValue = Math.max(...values, 1);
          const minValue = Math.min(...values, 0);
          const range = Math.max(maxValue - minValue, 0.1);

          const points = values.map((value, index) => {
            const x = index * ((w - 32) / (values.length - 1)) + 16;
            const y = h - 18 - ((value - minValue) / range) * (h - 36);
            return { x, y, value, index };
          });

          const gradient = ctx.createLinearGradient(0, 0, w, 0);
          gradient.addColorStop(0, '#f08a5d');
          gradient.addColorStop(1, '#d96842');

          const renderPath = (progress) => {
            ctx.beginPath();
            points.forEach((point, index) => {
              const drawUntil = Math.max(1, Math.ceil(progress * (points.length - 1)));
              if (index === 0) {
                ctx.moveTo(point.x, point.y);
              } else if (index <= drawUntil) {
                ctx.lineTo(point.x, point.y);
              }
            });
            ctx.lineWidth = 4;
            ctx.strokeStyle = gradient;
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';
            ctx.stroke();

            points.forEach((point) => {
              ctx.beginPath();
              ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
              ctx.fillStyle = '#d96842';
              ctx.fill();
            });
          };

          if (animate) {
            cancelAnimationFrame(animationFrameId);
            let progress = 0;
            const tick = () => {
              renderPath(progress);
              progress += 0.08;
              if (progress <= 1) {
                animationFrameId = requestAnimationFrame(tick);
              }
            };
            tick();
            return;
          }

          renderPath(1);

          const tooltip = $('chartTooltip');
          const canvasBounds = canvas.getBoundingClientRect();
          canvas.onmousemove = (event) => {
            const mouseX = event.clientX - canvasBounds.left;
            const mouseY = event.clientY - canvasBounds.top;
            let nearest = null;

            points.forEach((point) => {
              const distance = Math.hypot(mouseX - point.x, mouseY - point.y);
              if (!nearest || distance < nearest.distance) {
                nearest = { point, distance };
              }
            });

            if (!nearest || nearest.distance > 18) {
              tooltip.style.opacity = '0';
              return;
            }

            tooltip.style.opacity = '1';
            tooltip.style.left = `${event.clientX - canvasBounds.left + 10}px`;
            tooltip.style.top = `${event.clientY - canvasBounds.top - 12}px`;
            const valueColor = nearest.point.value >= 0.75 ? '#f08a5d' : nearest.point.value >= 0.45 ? '#d1a85b' : '#9cd7c2';
            tooltip.innerHTML = `<span style="color:${valueColor};">${nearest.point.value.toFixed(3)}</span>`;
          };

          canvas.onmouseleave = () => {
            tooltip.style.opacity = '0';
          };
        }

        function updateHistoryWithPrediction(stressValue) {
          dashboardHistory.push(Number(stressValue));
          if (dashboardHistory.length > 12) {
            dashboardHistory = dashboardHistory.slice(-12);
          }
          const value = Number(stressValue);
          $('metric').textContent = value.toFixed(3);
          setStatusIndicator('predicción', value);
          draw(dashboardHistory, true);
        }

        async function loadDashboard() {
          try {
            const healthResponse = await fetch('/health');
            const health = await healthResponse.json();
            const pill = $('health');
            pill.textContent = health.model_available ? 'API en línea' : 'Modelo no listo';
            pill.classList.toggle('offline', !health.model_available);

            const dataResponse = await fetch('/dashboard-data');
            const data = await dataResponse.json();
            const history = data.history ? data.history.map((item) => Number(item.value)) : [0.35, 0.41, 0.42];
            dashboardHistory = history;
            const value = Number(data.current_stress || history[history.length - 1] || 0).toFixed(2);
            $('metric').textContent = value;
            setStatusIndicator(data.status || 'normal', value);
            draw(history, false);
          } catch (error) {
            $('health').textContent = 'Sin conexión';
            $('health').classList.add('offline');
            $('metric').textContent = '--';
            $('state').textContent = 'No se pudo cargar el historial';
          }
        }

        document.getElementById('audio').addEventListener('change', (event) => {
          const file = event.target.files[0];
          $('file-name').textContent = file ? file.name : 'Sin archivos seleccionados';
        });

        document.getElementById('submit').addEventListener('click', async () => {
          const file = document.getElementById('audio').files[0];
          if (!file) {
            $('file-name').textContent = 'Primero selecciona un archivo .wav';
            return;
          }

          const button = $('submit');
          const panel = $('resultPanel');
          button.disabled = true;
          button.textContent = 'Procesando...';
          panel.classList.add('visible');
          $('resultValue').textContent = '--';
          $('resultMeta').textContent = 'Analizando audio...';

          const formData = new FormData();
          formData.append('file', file);

          try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();

            if (!response.ok) {
              throw new Error(data.detail || 'No se pudo procesar el archivo');
            }

            const stress = Number(data.stress_index || 0);
            const stressType = classifyStress(stress);
            const label = stressType.label;
            const color = stressType.className === 'danger' ? '#d96842' : stressType.className === 'alert' ? '#d1a85b' : '#4b8b73';

            $('resultValue').textContent = stress.toFixed(3);
            $('resultMeta').textContent = `Nivel estimado: ${label} · archivo ${file.name}`;
            $('resultValue').style.color = color;
            updateHistoryWithPrediction(stress);
          } catch (error) {
            $('resultValue').textContent = 'Error';
            $('resultMeta').textContent = error.message || 'No se pudo completar la predicción';
            $('resultValue').style.color = '#d96842';
          } finally {
            button.disabled = false;
            button.textContent = 'Ejecutar predicción';
          }
        });

        loadDashboard();
      </script>
    </body>
    </html>
    """


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=415, detail="El archivo debe ser .wav")
    spectrogram, laeq, lafmax = _audio_to_tensor(await file.read())
    model = _load_model()
    with torch.no_grad():
        stress_index = float(model(spectrogram).item())
    return {
      "filename": file.filename,
      "stress_index": stress_index,
      "laeq": laeq,
      "lafmax": lafmax,
      "status": "predicted",
    }


@app.get("/dashboard-data")
def dashboard_data() -> dict:
    return {
        "current_stress": 0.42,
        "status": "normal",
        "history": [
            {"timestamp": "2026-09-07T08:00:00Z", "value": 0.35},
            {"timestamp": "2026-09-07T10:00:00Z", "value": 0.41},
            {"timestamp": "2026-09-07T12:00:00Z", "value": 0.42},
        ],
    }
