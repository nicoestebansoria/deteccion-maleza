# Contexto del proyecto (para retomar)

Última actualización: 2026-11-02 (paso de Streamlit a app de escritorio).

## Qué es

Porte local del notebook de Colab de detección de malezas en ortomosaicos de dron
(`legacy/yolo_limpio.py`). Detecta con YOLOv8, poligoniza y exporta. Tiene una app de escritorio PySide6 (antes era Streamlit; esa versión quedó en
`legacy/app_streamlit.py`).
Ver [README.md](README.md) para instalación y uso.

## Entorno

- Venv con uv (`.venv`, Python 3.12). Torch en CPU: no hay CUDA (Intel Ultra 7 155H + Arc iGPU).
- Abrir la app: doble click en `abrir_app.bat`, o `.venv/Scripts/python -m maleza app`.
  El .bat usa pythonw (sin consola) y los mensajes van a `outputs/app.log`.
  Mientras corre un entrenamiento, `uv run` falla porque Windows bloquea `maleza.exe`: usar
  `uv run --no-sync` o el python del venv.
- `src/maleza/__init__.py` borra PROJ_LIB/GDAL_DATA de MapWinGIS (rompían rasterio).
  Si un script no importa `maleza` primero, aparecen errores de DLL de MapWinGIS.
- Repo público: https://github.com/nicoestebansoria/deteccion-maleza. Ramas: `dev` (donde se hacen las mejoras),
  `prod` (versión definitiva) y `main` (backup). De las corridas solo se versiona `runs/v4/weights/best.pt`; `data/`,
  `outputs/` y los zips quedan fuera del repo.

## Datos

- Ortomosaico de prueba: `data/orthos/viscon-rgb-orthophoto-recortado.tif`. Es 14151x9494, RGBA,
  EPSG:32720, 5 cm/px y el lote está en barbecho.
- `data/tiles`: 120 tiles reales + `tiles_metadata.json`, más 146 `tile_aug_*` viejos que se ignoran.
  `data/labels`: etiquetas actuales. `data/labels_backup_pre_revision`: copia de respaldo.
- En la raíz quedan tiles.zip y dos zips de labels, que se pueden borrar.

## Modelo

- **v4** (`runs/v4/weights/best.pt`) es el predeterminado en `config.yaml`. El v3 de Colab no está acá.
  - Entrenamiento: desde COCO, 114 épocas en 6.1 h, con corte anticipado. La mejor época fue la 84.
  - Validación con split espacial: P 0.71, R 0.52, mAP50 0.61, mAP50-95 0.24.
  - La validación es chica (21 tiles, 86 cajas): las métricas son ruidosas. El punto débil es el recall.
- Hallazgo: las etiquetas mezclan criterios. La mayoría son plantas individuales (mediana de 1.7 m
  de lado), pero hay 108 cajas que encierran un manchón entero, de más de 5 m y hasta 51 m. Por eso
  el modelo detecta los manchones grandes en partes. Pendiente: unificar el criterio de etiquetado.

## Pipeline actual de la app

Ventana Qt: panel izquierdo con pestañas Detección / Entrenamiento y mapa folium en un
QWebEngineView a la derecha. Los cálculos largos (vista previa, detección, polígonos) corren en hilos
(`Task`); los sliders recalculan con un debounce de 400 ms y el mapa se redibuja conservando el encuadre.

1. **Ortomosaico:** desplegable con `data/orthos` + `data/uploads`, o **Abrir…** desde cualquier carpeta.
   Debajo se muestran tamaño, resolución, bandas y CRS, con avisos si algo no cuadra.
2. **Detección:** sobre el .tif con ventana deslizante (1024 px, 200 de solape), con
   conf = `review_conf_min` (0.1). Las cajas duplicadas entre tiles se eliminan por solape:
   IoU, o IoS para cajas truncadas en el borde del tile.
3. **Resultado:** modelo + complemento por índice de verde. Está fijo; no hay selector de método.
   - `green_components`: ExG de todo el orto a resolución original (`green_resolution_m: 0`), borrado de
     motas < 0.04 m², cierre de 0.3 m y separación en manchas conexas. Calcula el área, el ExG medio y el
     ExG pico (sobre el ExG promediado a 10 cm) de cada mancha. La app lo cachea. Tarda ~8 s y ~3.7 GB.
   - Por qué resolución original: a 10 cm, las plantas chicas o pálidas se promediaban con el suelo y
     el modelo las detectaba pero quedaban "sin verde" (28 descartadas → 13 ahora; las 15 recuperadas
     son plantas reales, verificado con recortes). Sin el borrado de motas, a 5 cm el rastrojo deja
     píxeles sueltos que el cierre une en manchones falsos (el área se multiplicaba por 4). Un filtro de
     mediana también lo evitaba, pero borraba plantas chicas (32 descartadas).
   - `polygonize_green`: cada detección toma las manchas que toca, completas. Las detecciones que
     comparten mancha se unen en un polígono (`n_det`).
   - Opcionalmente suma manchas sin detección de 0.25 m² o más (`source=exg`, naranja).
   - Descarta las manchas con ExG pico < 0.13, que son rastrojo, y las detecciones sin verde.
     Este umbral se calibró con recortes de muestra: por debajo de 0.13 casi no hay plantas.
   - `green_max_expand_m = 0`: sin límite. Con el límite de 5 m aparecían cortes rectos en los manchones.
   - Todos los ajustes son sliders visibles. Los polígonos se recalculan solos; la máscara de verde
     se cachea (2 entradas) por orto + umbral + resolución + cierre.
   - Orto en el mapa: de lejos, una vista previa WebP de 2048 px (~35 cm/px) embebida en el HTML; desde
     zoom 18, teselas a resolución completa de `tiles.TileServer` (servidor HTTP local en 127.0.0.1,
     puerto aleatorio, lee el .tif con WarpedVRT a EPSG:3857). Ambas en un panel propio debajo de los
     polígonos. Una pantalla en zoom 21 carga en ~0.5 s.
   - El mapa folium muestra los polígonos (las cajas se prenden con un checkbox o el selector de capas). El orto se reproyecta
     a EPSG:3857 con WarpedVRT; sin eso quedaba corrido hasta 12 m.
   - El botón de exportar polígonos (GPKG/GeoJSON/SHP) se sacó de la app a pedido del usuario (2026-11-08);
     la salida es la prescripción. `polygonize.export` sigue para el CLI (`maleza predict`).
   - Resultado típico en el orto de prueba (conf 0.35): unos 2020 polígonos y unos 6220 m²
     (a 10 cm eran 1470 y 5050 m²).
4. **Pestaña Entrenamiento:** ver el pool de entrenamiento y el estado de las corridas, y
   reentrenar en segundo plano (`launch_training`, con un dataset por corrida en `data/datasets/<nombre>`).

## Calibración de ajustes (`maleza calibrar`, 2026-11-03)

`src/maleza/calibrate.py` mide la salida contra `data/labels` (749 cajas tras deduplicar, 120 tiles, 21 de
validación del v4). Recall = cajas tocadas por alguna mancha; precisión por área (por cantidad premiaba
manchones gigantes). Tabla en `outputs/<orto>/calibracion.csv`; detecciones crudas cacheadas al lado.

- Config actual: recall 0.955 (val 0.986), precisión 0.71 por área, 1074 manchas que no tocan etiquetas.
- Sin manchas sin detección, el recall cae a 0.52: el índice de verde hace la mayor parte del trabajo.
  La confianza mínima casi no cambia nada (0.94-0.97). El umbral ExG 0.07 baja el recall a 0.86 y 0.03
  une todo en manchones. El área mínima < 0.2 no influye.
- La palanca fuerte es el área mínima de manchas sin detección: 0.5 → recall 0.86 y 438 manchas sin etiqueta.
- **Las "manchas falsas" en su mayoría son plantas chicas reales sin etiquetar** (muestra de 16 recortes):
  las etiquetas cubren sobre todo plantas grandes. Para medir precisión de verdad hace falta revisar
  manchas sí/no (idea: modo revisión en la app que muestre manchas al azar).

- **Revisión rápida (TEMPORAL, pedido 2026-11-03):** pestaña "Revisión (temporal)" en la app,
  código en `src/maleza/revision_rapida.py`. Muestra 4 x 50 manchas sin etiqueta, estratificadas por
  ExG pico (0.08/0.11/0.14/0.18), para responder sí/no; guarda en `outputs/<orto>/revision_manchas.json`.
  `maleza calibrar` agrega prec_revisada / reales_est / falsas_est. Para sacarla: borrar el archivo y
  las 2 líneas marcadas `revision_rapida` en app.py.

- **Resultado de la revisión (192 respuestas, 74 sí):** el ExG pico separa bien. 0.08-0.11: 4% maleza;
  0.11-0.14: 17%; 0.14-0.18: 48%; >= 0.18: 88%. Por área: < 0.25 m² 15%, >= 1 m² 100%.
- **Config calibrada (aplicada):** conf 0.1, verde mínimo de manchas del modelo 0.08, verde mínimo de
  manchas sin detección 0.14 (slider/clave nueva `green_orphan_min_peak_exg`), resto igual.
  Contra la anterior: recall 0.977 vs 0.955 (val 0.958 vs 0.986, son 2 cajas), malezas chicas sin
  etiqueta estimadas 1565 vs 1228, manchas falsas estimadas en todo el orto 136 vs 244.
  Con 0.13 para las manchas sin detección: recall 0.98, val 0.986, pero 274 falsas.
  Las estimaciones solo valen para ExG 0.05 y cierre 0.3, la configuración con que se tomó la muestra.

- **Verde pico absoluto (2026-11-03, reemplaza lo anterior):** el usuario vio falsos positivos en el medio del
  lote sobre rastrojo puro. Causa: el ExG normalizado (2G-R-B)/(R+G+B) se dispara en las sombras oscuras
  del rastrojo (divisor chico). Con las 192 respuestas, el pico de 2G-R-B absoluto (escala 0-255, promediado a
  10 cm) separa sí/no con AUC 0.98, contra 0.90 del normalizado. Claves nuevas `green_min_peak: 25` (manchas
  del modelo) y `green_orphan_min_peak: 40` (sin detección); reemplazan `green_min_peak_exg*`. Columna
  exportada `verde_pico` (antes `exg_pico`). Calibración: recall 0.981 (val 0.986), ninguna mancha revisada
  como «no» pasa el filtro. En el orto de prueba: 1746 polígonos, 6073 m², 568 naranjas (antes 742).
  El umbral ExG normalizado (0.05) sigue usándose para armar las manchas.
- Recorte del borde del orto (`border_exclude_m`, slider): implementado pero en 0. El 31% del área está a
  < 2 m del borde, pero el usuario aclaró que sus falsos positivos no son esos.

- `runs/app_v1` (2026-11-03): fine-tuning desde v4 lanzado desde la app y **detenido a pedido en la época 15/60**.
  Usaba las mismas 120 etiquetas que v4, así que no aportaba. Reentrenar cuando haya etiquetas nuevas
  (otro ortomosaico, plantas chicas, criterio unificado). Ojo: su split de validación difiere del de v4.

## Archivos clave

- `src/maleza/app.py`: la app Qt (`MainWindow`, `TrainingTab`).
- `src/maleza/calibrate.py`: calibración de ajustes contra las etiquetas (`maleza calibrar`).
- `src/maleza/tiles.py`: servidor de teselas del orto a resolución completa.
- `src/maleza/lote.py`: reproyección a UTM de los ortos en grados (`reproject_to_utm`; en la app, `self.utm` y
  `prepare_ortho`, y `base_ortho()` devuelve la copia) y perímetro del lote (`read_lots`, `lot_perimeter`, `clip_ortho` por franjas
  de 1024 filas a GeoTIFF RGBA con deflate, `clip_to_perimeter`). En la app, `ortho()` devuelve el orto recortado si
  hay recorte aplicado (None mientras se recorta) y `base_ortho()` el elegido en la lista. `self.lots` es la capa
  cargada; `best_lot` preselecciona en el desplegable el lote que cubre el orto, y el recorte se aplica recién con el
  botón **Recortar** (`self.perim`). Un lote = un elemento del archivo. Si el lote no tiene datos en el orto,
  `clip_ortho` lo rechaza y la app avisa y vuelve al orto completo.
- `src/maleza/prescripcion.py`: zonas de prescripción (`build_prescription`), perfiles por monitor (`PROFILES`,
  desplegable en la app), adaptación de las zonas al generar (`adapt_to_monitor`) y export (`export_for_monitor`:
  `<lote>_<monitor>/` con la estructura del pendrive + `.zip`).
  - Shapefile en WGS84 (.prj de ESRI), una fila por polígono, con `RATE` (N 10.2) como único campo. El fondo va
    solo si se marca la casilla. Excepción: Pro 700 / IntelliView III-IV (`as_boundary`) copia el archivo de Avansys
    que funciona cargado como límite: solo maleza, un registro multiparte con huecos, `rx` N 5.4 = 0.
  - ISOXML v3.3 con grilla tipo 1: TZN 0 = fuera del lote, 1 = fondo, 2 = maleza. DDI 1 (L/ha, ×100
    mm³/m²) o DDI 6 (kg/ha, ×100 mg/m²). La primera celda del BIN es la esquina sudoeste.
  - DJI: carpeta `DJI/Shapefile` (límite) + `DJI/Rx` (GeoTIFF de dosis), copiando lo que exporta
    Pix4Dfields.
  - **Ningún perfil se probó todavía en un monitor ni en un dron real.** Probar con un archivo
    chico antes de usarlo en campo.
  - Config en `prescripcion:` de `config.yaml`.
  - **Formatos por monitor (carpetas de pendrive, límites, dosis 0, requisitos pendientes):** ver
    [docs/prescripciones_monitores.md](docs/prescripciones_monitores.md) (relevamiento 2026-11-06).
- `src/maleza/mapa.py`: vista previa del orto, `build_map` (folium) y listados de ortos/pesos.
- `src/maleza/polygonize.py`: `green_components`, `polygonize_green`, `polygonize` (GrabCut, que ya
  no se usa en la app) y `export`.
- `src/maleza/predict.py`: `detect_ortho`, `merge_tiles_nms`.
- `src/maleza/dataset.py`: split espacial por bloques geográficos de 3x3 tiles, con sintéticos solo en train.
- `src/maleza/review.py`: revisión manual y pool de entrenamiento. **Se sacó de la UI a pedido del
  usuario, para agregarla más adelante.** El código sigue ahí. `outputs/<orto>/revision.json`
  tiene 1 tile revisado.
- `config.yaml`: claves `polygonize.green_*` y `exg_threshold`. Para volver al modo anterior:
  `green_resolution_m: 0.1`.

## Pendientes / ideas

- Uso posterior: a los polígonos exportados se les va a hacer un buffer + disolve (se probó +10/−10 m en la
  app y se sacó; se van a probar otras opciones). Por eso un polígono chico de más o de menos importa poco.
  Lo que más pesa: no perder plantas enteras y no meter falsos positivos sueltos.

- Volver a agregar la revisión manual de otra forma. Antes se usaba streamlit-image-annotation;
  en Qt habría que hacer un editor de cajas propio (QGraphicsView).
- Unificar el criterio de etiquetado y reentrenar partiendo de v4 (fine-tuning).
- `predict.conf` está en 0.35. El usuario prefería umbrales bajos; podría bajarse a 0.1.
- Filtro opcional para excluir la franja verde del borde del lote o del ortomosaico.
- Probar otros ortomosaicos (o2, o3, o4 en `config.yaml`) y ortomosaicos con cultivo, donde
  "todo lo verde es maleza" deja de valer.
