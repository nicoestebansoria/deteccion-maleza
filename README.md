# Detección de malezas en ortomosaicos

Pipeline YOLOv8 + GrabCut para detectar y poligonizar malezas en ortomosaicos RGB de dron.
Porteado del notebook de Colab (original en [legacy/yolo_limpio.py](legacy/yolo_limpio.py)).

## Instalación

```powershell
uv sync                    # entorno base (torch CPU)
uv sync --extra label      # + Jupyter y BBoxWidget para etiquetar
uv sync --extra openvino   # + OpenVINO (inferencia más rápida en CPU/GPU Intel)
```

Si tenés un entorno de conda activo, uv avisa que ignora `VIRTUAL_ENV`. Es normal.

## Datos

Copiá desde Drive (ver comentario arriba en [config.yaml](config.yaml)):

```
data/orthos/   *.tif
data/tiles/    tile_*.jpg + tiles_metadata*.json
data/labels/   tile_*.txt
runs/          weed_detection_v*/weights/best.pt
```

Los `tile_aug_*` viejos de Drive se ignoran: los sintéticos se regeneran en cada `dataset`.

## App de escritorio (cargar un ortomosaico y detectar)

```powershell
uv run maleza app          # abre la ventana (PySide6 / Qt)
```

También se abre con doble click en `abrir_app.bat` (sin consola: los mensajes quedan en
`outputs/app.log`). Ese lanzador funciona aunque haya un
entrenamiento corriendo: en ese caso `uv run maleza app` falla, porque Windows bloquea
`maleza.exe` mientras se usa y uv no puede reinstalarlo (se evita con `uv run --no-sync maleza app`).

Panel de controles a la izquierda (pestañas **Detección** y **Entrenamiento**) y mapa a la derecha.
El mapa usa Leaflet y el satélite de Esri, así que necesita internet para el fondo.

1. **Ortomosaico:** se elige de `data/orthos` (o `data/uploads`) o con **Abrir…** desde cualquier carpeta
   (no se copia). La app avisa si la
   resolución difiere mucho de la de entrenamiento (5 cm/px). Si el CRS está en grados, reproyecta sola una
   copia a su zona UTM en `outputs/<orto>/<orto>_UTM.tif` y trabaja con ella; el original no se toca.
   - **Perímetro del lote** (opcional): **Cargar perímetro…** acepta SHP, GPKG, GeoJSON o KML, en
     cualquier CRS, sueltos o dentro de un ZIP con una sola capa. Cada elemento del archivo es un lote: si hay varios,
     aparece un desplegable (el nombre sale de la columna `Lote`, `Nombre` o `Name`) y queda elegido solo el lote
     que cubre el ortomosaico. Se puede cambiar a mano; el mapa muestra todos con su nombre y el elegido en celeste.
   - **Recortar** aplica el recorte: nada se recorta hasta tocarlo. El orto recortado queda en
     `outputs/<orto>/recortes/<orto>__<lote>.tif` y, si ya existe y está al día, se reutiliza. La detección, el verde
     y el mapa usan ese recorte, y las cajas y los polígonos finales se recortan exactamente al perímetro. Aplicar
     otro recorte descarta la detección anterior. **Quitar** vuelve al orto completo.
2. **Detección:** corre el modelo con un umbral bajo (`review_conf_min`), así después se puede
   filtrar por confianza sin volver a detectar.
3. **Resultado:** el mapa muestra directamente los polígonos, que combinan el modelo con el índice
   de verde (ExG):
   - Cada detección toma la mancha verde completa que toca, y las detecciones que caen sobre la
     misma mancha se unen en un solo polígono (`n_det` = cuántas). Así se completan los manchones
     que el modelo detecta en partes.
   - También se suman las manchas verdes sin detección (naranja, `source=exg`). Pensado para lotes
     en barbecho, donde todo lo verde es maleza.
   - Limpieza de falsos positivos: se descartan las manchas cuyo verde más intenso (`exg_pico`) no
     llega a `green_min_peak_exg` (rastrojo o suelo apenas verdoso) y las detecciones del modelo
     sin verde.
   - Todo se ajusta en la sección **Ajustes** y se aplica con **Aplicar ajustes** (el mapa conserva el zoom).
4. **Prescripción:** une los polígonos de maleza en zonas de aplicación y asigna las dosis.
   - **Buffer positivo / negativo:** buffer +X, disolver y buffer −Y.
   - **Fondo** (opcional): agrega el resto del lote como otra zona, con su propia dosis. El lote es el
     perímetro cargado o, si no hay, el área con datos del orto. Sin marcarlo, el archivo lleva solo la zona de
     maleza. El perfil Case IH Pro 700 / NH IntelliView III-IV se carga como límite y nunca lleva fondo.
   - **Dosis:** la tabla tiene la columna `rx` para "Con maleza" y "Fondo", en L/ha o kg/ha.
     **Generar prescripción** la muestra en el mapa (violeta = maleza, verde = fondo), con las ha y el
     producto total.
   - **Monitor:** el desplegable define el formato, la carpeta del pendrive y cómo se arman las zonas. Debajo se
     muestra lo que aplica y lo que hay que configurar en el monitor. Cambiar de monitor regenera la prescripción.
     Los que necesitan un conversor o no tienen documentación aparecen deshabilitados. Detalle y fuentes en
     [docs/prescripciones_monitores.md](docs/prescripciones_monitores.md). **Ningún perfil se probó todavía en un
     monitor real: probar con agua y un archivo chico.**
   - **Grilla cuadrada** (casilla, apagada por defecto): pasa la maleza a bloques alineados al norte, del tamaño de
     celda elegido. Toda celda que toca maleza se aplica entera, así que aumenta la superficie aplicada.
   - **Manchón mínimo** (casilla, apagada por defecto): ancho de sección × velocidad × retardo. Los manchones más
     chicos se agrandan hasta esa superficie.
   - **Exportar prescripción** propone como nombre el del lote (o el del orto, sin recorte) y deja cambiarlo; los
     caracteres raros se reemplazan, porque algunos monitores no los leen. Crea `<nombre>_<monitor>/`, con la
     estructura lista para copiar a la raíz del pendrive, y el mismo contenido en un `.zip` (por defecto en
     `outputs/<orto>/prescripcion/`):
     - **Shapefile** en WGS84, una fila por polígono, con `RATE` como único campo.
     - **ISOXML** (`TASKDATA/TASKDATA.XML` + grilla tipo 1).
     - **DJI Agras:** carpeta `DJI/` con el límite (`Shapefile/`) y la dosis en GeoTIFF (`Rx/`).
   - La grilla de ISOXML y del TIFF usa celdas de `prescripcion.celda_m` (1 m). Una celda que toca
     maleza se aplica completa, así que el área aplicada sale algo mayor que la de las zonas.
5. **Reentrenar (pestaña Entrenamiento):** arma un dataset nuevo en `data/datasets/<nombre>` con todo el
   pool y entrena en segundo plano. Sigue aunque cierres la app y muestra el progreso de cada
   corrida.

La revisión manual de tiles está desactivada en la app por ahora. El código sigue en
[review.py](src/maleza/review.py).

## Línea de comandos

```powershell
uv run maleza info                      # resolución / CRS de los ortomosaicos
uv run maleza tile o2 o3                # generar tiles para etiquetar (vacío = todos)
uv run jupyter lab notebooks/           # etiquetar / revisar (etiquetado.ipynb)
uv run maleza dataset                   # split espacial + sintéticos + data.yaml
uv run maleza train --name v4           # entrenar desde COCO
uv run maleza calibrar                  # medir los ajustes de la app contra las etiquetas a mano
uv run maleza train --name v4b --weights runs/v4/weights/best.pt   # continuar
uv run maleza predict                   # todos los ortos del config
uv run maleza predict o2 --weights runs/v4/weights/best.pt
uv run maleza predict --tif D:/vuelos/nuevo.tif
uv run maleza inspect data/tiles/tile_00003.jpg    # confianzas crudas (diagnóstico)
```

Los resultados de `maleza predict` quedan en `outputs/<ortomosaico>/`: detecciones (cajas) y polígonos en GPKG,
y también GeoJSON y SHP.

## Cambios respecto al notebook

- **Split espacial**: los tiles se agrupan en bloques geográficos (3x3 tiles) y cada bloque va
  entero a train o a val. Antes, tiles solapados 200 px quedaban uno en train y otro en val, y eso
  inflaba las métricas. Como los bloques son por coordenadas, tiles de distintos vuelos sobre la
  misma zona también quedan juntos.
- **Poligonización sin pérdidas**: GrabCut fallaba en casi la mitad de las detecciones (maleza y
  rastrojo de color parecido) y esas malezas desaparecían de la salida. Ahora hay respaldo por ExG
  y, si todo falla, se usa la caja.
- **Sintéticos solo en train**, generados con donantes de train. Antes entraban a val.
- **Cajas sintéticas ajustadas a la planta**: antes el label era todo el recorte, con un margen
  del 35% de la diagonal. Además se agregaron flips verticales y rotaciones de 90°.
- **Faltaban `all_donors` y `boxes_overlap`** en la celda 18b: ahora están definidos.
- **Inferencia directa sobre el .tif**: se recorre con ventana deslizante, así que no hacen falta
  JPGs ni metadata y funciona con cualquier ortomosaico.
- **Dedup por solape de cajas en vez de DBSCAN**: con `min_samples=1`, DBSCAN encadenaba plantas
  vecinas de una fila en una sola detección. Ahora: IoU > 0.5, o caja cortada por el borde del
  tile contenida en otra (gana la completa).
- **Exporta también las cajas** (`malezas_detecciones.gpkg`), útil para comparar con los polígonos.

## Entrenar en esta máquina

No hay GPU NVIDIA (la Intel Arc no usa CUDA), así que el entrenamiento corre en CPU. Con
~120 tiles reales + sintéticos a 1024 px, calculá varios minutos por época. Opciones:
- entrenar en CPU de noche, o bajar `imgsz` / `epochs` para iterar rápido;
- seguir entrenando en Colab (`uv run maleza dataset` y subir `data/dataset/`);
- para inferencia: `yolo export model=best.pt format=openvino` y en `predict.weights` poner
  la carpeta `best_openvino_model/`, con `device: intel:gpu` o `cpu`.
