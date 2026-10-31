# Monitoreo y Predicción del Estrés Acústico con CNN-Transformer

Proyecto de tesis — Escuela Profesional de Ingeniería de Sistemas.

**Título:** Modelo híbrido CNN-Transformer para el monitoreo y la predicción del índice de estrés acústico por ruido no estacionario en el distrito de Ate, 2026.

**Autores:** Julca Herrera, Angel Gabriel · Lescano Icochea, José Luis
**Asesor:** Saboya Rios, Nemias

## Objetivo

Desarrollar una arquitectura híbrida CNN-Transformer que permita monitorear y predecir el índice de estrés acústico generado por ruido no estacionario en un entorno urbano-industrial (Av. Separadora Industrial, Ate), a partir de espectrogramas de Gabor.

## Arquitectura del modelo

```
Audio (.wav)
  -> Preprocesamiento (normalización + ventaneo)
  -> Espectrograma de Gabor (representación tiempo-frecuencia)
  -> Bloque CNN (extracción de características locales)
  -> Bloque Transformer Encoder (dependencias temporales de largo alcance)
  -> Cabeza de regresión (predicción del índice de estrés acústico)
```

## Estructura del proyecto

```
estres-acustico-cnn-transformer/
├── data/               # raw, interim, processed, external (no versionado en Git)
├── notebooks/          # exploración y experimentos (Colab / Kaggle / local)
├── src/
│   ├── config.py        # rutas e hiperparámetros centralizados
│   ├── data/             # carga de datos y generación de espectrogramas de Gabor
│   ├── models/            # bloques CNN, Transformer y modelo híbrido
│   ├── training/            # loop de entrenamiento y métricas
│   └── evaluation/            # evaluación y comparación con métodos tradicionales
├── models_saved/       # checkpoints entrenados (no versionado)
├── reports/            # figuras y bitácora de resultados
├── docs/                # bibliografía (BibTeX) para Overleaf
└── tests/               # pruebas unitarias
```

## Instalación

### Opción A: pip + venv

```bash
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Opción B: Conda

```bash
conda env create -f environment.yml
conda activate estres-acustico
```

## Cómputo

El entrenamiento está pensado para correr en GPU gratuita:

- **Google Colab** (T4 gratuita, sesiones limitadas)
- **Kaggle Notebooks** (30 h/semana de GPU gratis)

No se requiere GPU local.

## Datos

- Dataset base: [Monitoreo de ruido por zonas en el distrito de Ate (2019-2022)](https://www.datosabiertos.gob.pe/dataset/monitoreo-de-ruido-por-zonas-en-el-distrito-de-ate-desde-2019-al-2022-mda) — Plataforma Nacional de Datos Abiertos del Perú.
- Datasets complementarios de referencia: UrbanSound8K, DCASE Challenge.

Coloca los archivos de audio crudos en `data/raw/` (esta carpeta no se versiona en Git; ver `.gitignore`).

## Ejecución de pruebas

```bash
pytest tests/
```

## Dashboard local

Después de entrenar el modelo y generar `models_saved/best_model.pt`, inicia la API:

```bash
uvicorn api.main:app --reload
```

Abre `http://127.0.0.1:8000/` para usar el dashboard. Permite consultar el estado,
ver el historial de ejemplo y enviar archivos `.wav` al endpoint `/predict`. El audio
debe estar a 22050 Hz; la API aplica el mismo espectrograma de Gabor y normalización
que el pipeline de entrenamiento. Si todavía no existe el checkpoint, la interfaz
arranca pero `/predict` responde `503` indicando que primero se debe entrenar el modelo.

## Licencia

Ver archivo `LICENSE`.
