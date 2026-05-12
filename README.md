# mandarine

Repositorio del TFG de Pablo Ferrer Gonzalez sobre deteccion de anomalias visuales con `anomalib`, conservado en dos capas:

- una capa historica que documenta el trabajo entregado hace 3 años;
- una replica moderna en `Python 3.12` y `anomalib 2.4.1` para volver a entrenar y evaluar el caso de las mandarinas.

## Que contiene el proyecto

El TFG original estudiaba deteccion de anomalias en imagenes industriales, comparaba modelos inspirados en `MVTec AD` y cerraba con un caso practico IoT basado en imagenes de mandarinas y una simulacion con `PYNQ-Z2`. La evidencia historica mas fuerte apunta a `anomalib==0.3.7`.

La replica moderna no reutiliza el codigo antiguo de entrenamiento. En su lugar, levanta una estructura actual con `src/`, configs, pruebas, notebooks de reporte y artefactos reproducibles.

## Estructura

- `data/`: datasets y muestras de inferencia.
- `docs/`: memoria del TFG, presentacion y HTML exportados.
- `legacy/`: material historico archivado, incluida la copia parcial antigua de `anomalib`.
- `notebooks/`: notebooks historicos y dos notebooks modernos de reporte.
- `src/mandarine/`: pipeline moderno para preparar splits, entrenar, evaluar, inferir y reportar.
- `configs/modern/`: configuraciones YAML de dataset, modelos, benchmark y entrenamiento final.
- `tests/`: comprobaciones ligeras de splits y configs.
- `artifacts/modern/`: artefactos ligeros y publicables de la replica moderna.

## Datos

La fuente canonica del experimento moderno es `data/mandarins_pynq_cropped`, con:

- `33` imagenes normales
- `7` imagenes anomalas

Tambien se conservan:

- `data/mandarins_pynq_raw` para ejemplos cualitativos y contexto historico.
- `data/mandarins_pynq_augmented` como legado, sin usar en el pipeline moderno para evitar leakage.
- `data/webcam_inference_images` para demos de inferencia.

## Capa historica

La reconstruccion historica queda pensada para consulta y para reejecucion puntual de notebooks:

- version historica de referencia: `anomalib 0.3.7`
- entorno objetivo: `Python 3.8`
- dependencias: `requirements-legacy.txt`

La antigua carpeta `anomalib/` del repo se ha movido a `legacy/anomalib_snapshot/` para que la replica moderna use siempre el paquete instalado desde PyPI y no una copia parcial local.

## Replica moderna

La replica moderna usa:

- `Python 3.12`
- `anomalib 2.4.1`
- `PatchCore`, `EfficientAd` y `AnomalyDINO`
- evaluacion a nivel de imagen con `image_AUROC`, `image_AUPR`, `image_F1` y latencia media por imagen

### Protocolo

- dataset base: `data/mandarins_pynq_cropped`
- semillas fijas: `13`, `23`, `42`
- split por semilla:
  - normales: `23 train`, `5 val`, `5 test`
  - anomalas: `2 val`, `5 test`
- nunca se entrena con imagenes anomalas
- las augmentations solo se aplican on-the-fly al train normal
- la seleccion del ganador se hace por:
  - mayor `mean image_AUROC`
  - desempate por `mean image_AUPR`
  - segundo desempate por menor latencia

### Configuracion moderna

- `configs/modern/dataset/mandarins_cropped.yaml`: fuente de datos, seeds y tamanos de split.
- `configs/modern/models/*.yaml`: hiperparametros y `max_epochs` por modelo.
- `configs/modern/benchmark_cpu.yaml`: benchmark CPU-first sobre los 3 modelos.
- `configs/modern/final_train.yaml`: reentrenamiento final del ganador sobre la semilla `42`.

## Instalacion

### Entorno historico

```powershell
py -3.8 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-legacy.txt
python -m pip install anomalib==0.3.7 --no-deps
```

### Entorno moderno

```powershell
py -3.12 -m venv .venv-modern
.venv-modern\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-modern.txt
```

`requirements-modern.txt` instala tambien el paquete local en editable (`-e .`), asi que los comandos `python -m mandarine...` quedan disponibles sin tocar `PYTHONPATH`.

## Flujo moderno

### 1. Preparar los splits

```powershell
python -m mandarine.data.prepare
```

Genera los directorios derivados por semilla dentro de `artifacts/modern/splits/` y un `split_manifest.csv`.

### 2. Ejecutar el benchmark

```powershell
python -m mandarine.experiments.benchmark
```

Produce:

- `artifacts/modern/benchmark/benchmark_runs.csv`
- `artifacts/modern/benchmark/leaderboard.csv`
- `artifacts/modern/benchmark/metrics_summary.json`
- `artifacts/modern/benchmark/split_manifest.csv`

Si quieres una pasada mas ligera en CPU, puedes filtrar modelos o seeds:

```powershell
python -m mandarine.experiments.benchmark --models patchcore anomalydino
python -m mandarine.experiments.benchmark --models patchcore --seeds 42
```

### 3. Reentrenar el ganador

```powershell
python -m mandarine.experiments.train_final
```

Guarda el resumen final en `artifacts/modern/final_model/final_model_summary.json`.

### 4. Inferencia sobre la carpeta de webcam

```powershell
python -m mandarine.inference.predict_folder
```

Genera tablas de prediccion en `artifacts/modern/predictions/`.

### 5. Reporte y HTML

```powershell
python -m mandarine.reporting.build_report
```

Este paso:

- genera figuras comparativas;
- ejecuta los notebooks modernos;
- exporta HTML a `docs/html/modern/`.

## Notebooks

### Historicos

- `notebooks/tfg_experiments.ipynb`
- `notebooks/model_boosting_experiments.ipynb`
- `notebooks/iot_orange_experiments.ipynb`

Siguen siendo cuadernos de contexto y reconstruccion.

### Modernos

- `notebooks/modern_benchmark_report.ipynb`: resume leaderboard, runs y criterio de seleccion del modelo ganador.
- `notebooks/modern_inference_demo.ipynb`: resume el entrenamiento final y las predicciones sobre las imagenes de webcam.

Los notebooks modernos leen artefactos ya generados; no entrenan ellos mismos.

## Que se versiona de la replica moderna

Para que el repositorio sea publicable y siga siendo ligero, se conservan solo artefactos de salida pequenos y utiles para inspeccion:

- `artifacts/modern/benchmark/*.csv` y `metrics_summary.json`
- `artifacts/modern/benchmark/figures/*.png`
- `artifacts/modern/final_model/final_model_summary.json`
- `artifacts/modern/predictions/*.csv`, `*.json` y `predictions/visualizations/*.png`

Los directorios de entrenamiento, checkpoints, caches y copias repetidas de imagenes se regeneran al ejecutar el pipeline moderno y no se versionan.

## Pruebas

```powershell
pytest
```

Las pruebas cubren:

- determinismo de splits
- conteos esperados por seed
- ausencia de anomalias en train
- carga de configs y resolucion de rutas

## Limitaciones conocidas

- `EfficientAd` descarga `Imagenette` y pesos auxiliares en el entorno moderno, por lo que es el modelo mas pesado de preparar en esta maquina.
- El benchmark actual esta planteado para CPU porque esta maquina no expone GPU visible.
- El dataset de mandarinas no tiene mascaras, asi que la evaluacion moderna se centra en clasificacion a nivel de imagen y no en segmentacion.

## Documentacion base

- `docs/tfg_memoria.pdf`
- `docs/tfg_memoria.full.txt`
- `docs/tfg_presentacion.pdf`
- `docs/tfg_presentacion.full.txt`
- `docs/reconstruction_notes.md`

## Licencia

El codigo del repositorio se publica bajo licencia `MIT`. Los documentos del TFG y materiales historicos se conservan como contexto academico del proyecto.
