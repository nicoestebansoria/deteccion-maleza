"""App de escritorio: cargar un ortomosaico, detectar malezas (modelo + complemento por índice de verde),
ver los polígonos, exportarlos y reentrenar.

Lanzar con:  uv run maleza app   (o doble click en abrir_app.bat)
"""
import base64
import json
import math
import os
import sys
import tempfile
import threading
import traceback
from collections import OrderedDict
from pathlib import Path

# torch antes que Qt: en Windows, si Qt carga primero sus DLLs, torch puede fallar al inicializarse
import torch  # noqa: F401
import rasterio
from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                               QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QRadioButton, QScrollArea, QSlider, QSpinBox,
                               QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from maleza import review as rv
from maleza.lote import (PERIMETER_FILTER, clip_ortho, clip_to_perimeter, clipped_path, is_fresh, lot_perimeter,
                         perimeter_in, read_lots, reproject_to_utm, utm_path)
from maleza.mapa import available_weights, build_map, list_orthos, ortho_info, ortho_preview
from maleza.polygonize import clip_border, export, green_components, ortho_footprint, polygonize_green
from maleza.predict import detect_ortho, load_model
from maleza.prescripcion import FORMATS, UNITS, build_prescription, export_prescription, safe_name, summary
from maleza.tiles import TileServer
from maleza.train import launch_training, run_status

RUNS_REFRESH_MS = 30_000     # refresco del estado de los entrenamientos

STYLE = """
QFrame#card {border: 1px solid rgba(128,128,128,.35); border-left: 6px solid #2e7d32; border-radius: 6px;}
QLabel#cardLbl {color: #8a8a8a; font-size: 9pt;}
QLabel#cardVal {font-size: 16pt;}
QLabel#warn {color: #b26a00;}
QLabel#error {color: #c62828;}
QLabel#hint {color: #8a8a8a;}
"""


class Task(QObject):
    """Corre fn(emit_progress) en un hilo aparte y avisa por señales (que llegan al hilo de la UI)."""
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            self.done.emit(self.fn(self.progress.emit))
        except Exception:
            self.failed.emit(traceback.format_exc())


class FloatSlider(QWidget):
    valueChanged = Signal(float)

    def __init__(self, label, lo, hi, step, value, tip=""):
        super().__init__()
        self.lo, self.step = lo, step
        self.decimals = max(0, -math.floor(math.log10(step)))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, round((hi - lo) / step))
        self.slider.setValue(round((min(max(value, lo), hi) - lo) / step))
        self.val = QLabel()
        title = QLabel(label)
        title.setWordWrap(True)
        head = QHBoxLayout()
        head.addWidget(title, 1)
        head.addWidget(self.val)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.addLayout(head)
        lay.addWidget(self.slider)
        self.setToolTip(tip)
        self.slider.valueChanged.connect(self._changed)
        self._changed()

    def value(self):
        return round(self.lo + self.slider.value() * self.step, 6)

    def _changed(self, *_):
        self.val.setText(f"{self.value():.{self.decimals}f}")
        self.valueChanged.emit(self.value())


class Card(QFrame):
    def __init__(self, label, tip=""):
        super().__init__()
        self.setObjectName("card")
        self.setToolTip(tip)
        lbl, self.val = QLabel(label), QLabel("—")
        lbl.setObjectName("cardLbl")
        self.val.setObjectName("cardVal")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.addWidget(lbl)
        lay.addWidget(self.val)

    def set(self, value):
        self.val.setText(str(value))


def label(text, kind=None):
    w = QLabel(text)
    w.setWordWrap(True)
    if kind:
        w.setObjectName(kind)
    return w


class TrainingTab(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win, self.cfg = win, win.cfg
        lay = QVBoxLayout(self)
        lay.addWidget(label("Tiles etiquetados en el pool (sin contar sintéticos):"))
        self.pool = QTableWidget(0, 2)
        self.pool.setHorizontalHeaderLabels(["Ortomosaico", "Tiles"])
        self.pool.verticalHeader().hide()
        self.pool.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.pool.setEditTriggers(QTableWidget.NoEditTriggers)
        self.pool.setMaximumHeight(160)
        lay.addWidget(self.pool)

        lay.addWidget(label("<b>Corridas</b>"))
        self.runs_lbl = label("")
        self.runs_lbl.setTextFormat(Qt.RichText)
        lay.addWidget(self.runs_lbl)

        box = QGroupBox("Reentrenar con todo el pool")
        form = QVBoxLayout(box)
        self.name = QLineEdit()
        form.addWidget(QLabel("Nombre de la corrida"))
        form.addWidget(self.name)
        self.base_model = QRadioButton("Modelo seleccionado (fine-tuning, más rápido)")
        self.base_coco = QRadioButton("Pesos COCO (desde cero)")
        self.base_model.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.base_model)
        group.addButton(self.base_coco)
        form.addWidget(QLabel("Partir de"))
        form.addWidget(self.base_model)
        form.addWidget(self.base_coco)
        self.epochs = QSpinBox()
        self.epochs.setRange(10, 500)
        self.epochs.setSingleStep(10)
        self.epochs.setValue(60)
        row = QHBoxLayout()
        row.addWidget(QLabel("Épocas"))
        row.addWidget(self.epochs, 1)
        form.addLayout(row)
        form.addWidget(label("Arma un dataset nuevo en data/datasets/<nombre> (split espacial + sintéticos) y "
                             "entrena en segundo plano: sigue aunque cierres la app. En CPU ~3 min por época.",
                             "hint"))
        self.running_warn = label("", "warn")
        self.force = QCheckBox("Lanzar igual")
        form.addWidget(self.running_warn)
        form.addWidget(self.force)
        self.launch_btn = QPushButton("Lanzar entrenamiento")
        form.addWidget(self.launch_btn)
        lay.addWidget(box)
        lay.addStretch(1)

        self.base_model.toggled.connect(lambda on: self.epochs.setValue(60 if on else 150))
        self.name.textChanged.connect(self._update_button)
        self.force.toggled.connect(self._update_button)
        self.launch_btn.clicked.connect(self.launch)
        self.existing, self.running = set(), []
        self.refresh()
        timer = QTimer(self)
        timer.timeout.connect(self.refresh)
        timer.start(RUNS_REFRESH_MS)

    def refresh(self):
        pool = rv.training_pool_summary(self.cfg)
        self.pool.setRowCount(len(pool))
        for i, (orto, n) in enumerate(pool.items()):
            self.pool.setItem(i, 0, QTableWidgetItem(str(orto)))
            self.pool.setItem(i, 1, QTableWidgetItem(str(n)))

        runs = run_status(self.cfg["paths"]["runs_dir"])
        self.running = [r for r in runs if r["running"]]
        lines = []
        for r in runs[-6:]:
            estado = "🟢 entrenando" if r["running"] else "✔ terminado"
            map50 = f" · mAP50 {r['map50_best']:.3f}" if r["map50_best"] is not None else ""
            lines.append(f"<b>{r['name']}</b> — {estado} · época {r['epoch']}/{r['epochs']}{map50}")
        self.runs_lbl.setText("<br>".join(lines) or "Sin corridas.")

        old_existing, self.existing = self.existing, {r["name"] for r in runs}
        if not self.name.text() or self.name.text() in self.existing - old_existing:
            n = 1
            while f"app_v{n}" in self.existing:
                n += 1
            self.name.setText(f"app_v{n}")
        if self.running:
            self.running_warn.setText(f"Ya hay un entrenamiento en curso ({', '.join(r['name'] for r in self.running)}). "
                                      "Dos a la vez compiten por la CPU y tardan el doble.")
        self.running_warn.setVisible(bool(self.running))
        self.force.setVisible(bool(self.running))
        self._update_button()

    def _update_button(self, *_):
        name = self.name.text().strip()
        self.launch_btn.setEnabled(bool(name) and name not in self.existing
                                   and (not self.running or self.force.isChecked()))

    def launch(self):
        weights = self.win.weights() if self.base_model.isChecked() else None
        log = launch_training(self.win.config_path, self.name.text().strip(), weights=weights,
                              epochs=self.epochs.value())
        QMessageBox.information(self, "Entrenamiento", f"Entrenamiento lanzado.\nLog: {log}")
        self.force.setChecked(False)
        self.refresh()


class MainWindow(QMainWindow):
    def __init__(self, cfg, config_path):
        super().__init__()
        self.cfg, self.config_path = cfg, str(config_path)
        self.tmp = tempfile.TemporaryDirectory(prefix="maleza_", ignore_cleanup_errors=True)
        self._tasks = set()
        self.models = {}
        self.tiles = TileServer()          # orto a resolución completa en el mapa al acercarse
        self.footprints = {}               # orto -> polígono del área con datos (para excluir el borde)
        self.green_cache = OrderedDict()   # (orto, umbral, res, cierre, motas, ventana pico) -> green_components, máx. 2
        self.previews = {}                 # orto -> (data URI de la vista previa, bounds)
        self.info = None
        self.utm = {}                      # orto elegido -> con cuál se trabaja: él mismo, o su copia en UTM si venía
        #                                    en grados (None mientras se reproyecta)
        self.perim = None                  # {"path", "gdf", "recorte"} perímetro con el que se recorta
        self.lots = None                   # {"path", "gdf"} capa con más de un lote, para elegir uno
        self.clips = {}                    # (orto, recorte) -> orto recortado al lote
        self.clipping = None               # (orto, perímetro) que se está recortando
        self.det = None                    # {"ortho", "weights", "raw", "n_raw"}
        self.final = self.polys = None
        self.rx = self.rx_boundary = None  # prescripción generada y límite del lote que usó
        self.poly_busy = self.poly_pending = False
        self.map_name, self.map_n = None, 0

        self.setWindowTitle("Detección de malezas en ortomosaicos")
        self.resize(1500, 920)

        tabs = QTabWidget()
        tabs.addTab(self._scroll(self._build_detection_panel()), "Detección")
        tabs.addTab(self._scroll(TrainingTab(self)), "Entrenamiento")
        from maleza.revision_rapida import ReviewTab  # TEMPORAL: revision_rapida
        tabs.addTab(self._scroll(ReviewTab(self)), "Revisión (temporal)")
        tabs.setMinimumWidth(380)

        self.view = QWebEngineView()
        # el mapa es un .html local que carga Leaflet y el satélite de internet
        self.view.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        self.view.setHtml("<body style='font-family:sans-serif;color:#777;padding:2em'>"
                          "Elegí un ortomosaico para ver el mapa.</body>")

        split = QSplitter()
        split.addWidget(tabs)
        split.addWidget(self.view)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 1080])
        self.setCentralWidget(split)

        self._load_lists()

    # ---------- armado de la UI ----------

    @staticmethod
    def _scroll(widget):
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(widget)
        return area

    def _build_detection_panel(self):
        panel = QWidget()
        lay = QVBoxLayout(panel)

        box = QGroupBox("Modelo")
        bl = QVBoxLayout(box)
        self.weights_combo = QComboBox()
        bl.addWidget(self.weights_combo)
        self.weights_msg = label("", "error")
        self.weights_msg.hide()
        bl.addWidget(self.weights_msg)
        lay.addWidget(box)

        box = QGroupBox("1. Ortomosaico")
        bl = QVBoxLayout(box)
        row = QHBoxLayout()
        self.ortho_combo = QComboBox()
        self.ortho_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.open_btn = QPushButton("Abrir…")
        self.open_btn.setToolTip("Elegir un GeoTIFF de cualquier carpeta (no se copia)")
        row.addWidget(self.ortho_combo, 1)
        row.addWidget(self.open_btn)
        bl.addLayout(row)
        self.info_form = QFormLayout()
        self.info_lbls = {k: QLabel("—") for k in ("Tamaño", "Resolución", "Bandas", "CRS")}
        for k, w in self.info_lbls.items():
            w.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.info_form.addRow(k + ":", w)
        bl.addLayout(self.info_form)
        self.ortho_msg = label("", "warn")
        self.ortho_msg.hide()
        bl.addWidget(self.ortho_msg)
        bl.addWidget(label("<b>Perímetro del lote</b>"))
        row = QHBoxLayout()
        self.perim_btn = QPushButton("Cargar perímetro…")
        self.perim_btn.setToolTip("ZIP, SHP, GPKG, GeoJSON o KML con el contorno del lote. Se recorta el ortomosaico "
                                  "y los polígonos finales a ese contorno.")
        self.perim_clear = QPushButton("Quitar")
        self.perim_clear.setEnabled(False)
        row.addWidget(self.perim_btn, 1)
        row.addWidget(self.perim_clear)
        bl.addLayout(row)
        self.perim_lbl = label("Sin perímetro: se usa todo el ortomosaico.", "hint")
        bl.addWidget(self.perim_lbl)
        self.lots_combo = QComboBox()  # solo si la capa tiene más de un lote
        self.lots_combo.hide()
        bl.addWidget(self.lots_combo)
        self.clip_progress = QProgressBar()
        self.clip_progress.hide()
        bl.addWidget(self.clip_progress)
        lay.addWidget(box)

        box = QGroupBox("2. Detección")
        bl = QVBoxLayout(box)
        self.detect_btn = QPushButton("Detectar malezas")
        self.detect_btn.setDefault(True)
        self.progress = QProgressBar()
        self.progress.hide()
        bl.addWidget(self.detect_btn)
        bl.addWidget(self.progress)
        lay.addWidget(box)

        pc = self.cfg["polygonize"]
        self.settings_box = box = QGroupBox("3. Ajustes")
        bl = QVBoxLayout(box)
        self.s_conf = FloatSlider("Confianza mínima del modelo", 0.05, 0.95, 0.05, self.cfg["predict"]["conf"],
                                  "Filtra las detecciones en vivo sin volver a detectar.")
        self.s_exg = FloatSlider("Umbral de verde (ExG)", 0.0, 0.2, 0.01, pc["exg_threshold"],
                                 "Más bajo = toma verdes más pálidos (y más rastrojo). Más alto = solo verde intenso.")
        self.c_orphans = QCheckBox("Agregar manchas verdes sin detección")
        self.c_orphans.setChecked(True)
        self.c_orphans.setToolTip("Manchas que el modelo no marcó: salen en naranja con origen 'exg' y confianza 0. "
                                  "Incluye cualquier verde (también bordes del lote).")
        self.s_orphan_area = FloatSlider("Área mínima de manchas sin detección (m²)", 0.05, 5.0, 0.05,
                                         pc["green_orphan_min_m2"],
                                         "Más bajo = suma plantas chicas sueltas (y más ruido).")
        self.s_orphan_peak = FloatSlider("Verde mínimo de manchas sin detección (pico 2G−R−B)", 0, 100, 1,
                                         pc.get("green_orphan_min_peak", 40),
                                         "Sin el modelo que las confirme, las manchas de verde tenue suelen ser rastrojo. "
                                         "Más alto = menos manchas naranjas falsas (y se pierden plantas pálidas).")
        self.s_peak = FloatSlider("Verde mínimo de manchas del modelo (pico 2G−R−B)", 0, 100, 1,
                                  pc.get("green_min_peak", 25),
                                  "Descarta manchas tocadas por el modelo cuyo píxel más verde no llega a este valor. "
                                  "Puede ser bajo: el modelo ya confirma que hay una planta.")
        self.s_area = FloatSlider("Área mínima de mancha (m²)", 0.02, 1.0, 0.02, pc.get("green_min_area_m2", 0.1),
                                  "Manchas más chicas se ignoran, tengan o no detección.")
        self.s_border = FloatSlider("Excluir franja del borde del ortomosaico (m)", 0.0, 20.0, 0.5,
                                    pc.get("border_exclude_m", 0.0),
                                    "Recorta los polígonos a esta distancia del borde del orto: alambrado, banquina y "
                                    "cultivo vecino son verde real pero no maleza del lote. 0 = sin recorte.")
        self.c_keep = QCheckBox("Conservar detecciones del modelo sin verde")
        self.c_keep.setToolTip("Detecciones sin ninguna mancha verde válida. En barbecho suelen ser falsos "
                               "positivos; activado, se poligonizan con GrabCut.")
        for w in (self.s_conf, self.s_exg, self.c_orphans, self.s_orphan_area, self.s_orphan_peak):
            bl.addWidget(w)
        bl.addWidget(label("<b>Limpieza de falsos positivos</b>"))
        for w in (self.s_border, self.s_peak, self.s_area, self.c_keep):
            bl.addWidget(w)
        # los sliders no recalculan solos: se ajustan todos y se aplica con el botón
        self.apply_btn = QPushButton("Aplicar ajustes")
        self.apply_btn.setMinimumHeight(34)
        self.apply_btn.setEnabled(False)
        bl.addWidget(self.apply_btn)
        lay.addWidget(box)

        rc = self.cfg.get("prescripcion", {})
        self.rx_box = box = QGroupBox("4. Prescripción")
        bl = QVBoxLayout(box)
        self.s_buf_pos = FloatSlider("Buffer positivo (m)", 0.0, 20.0, 0.5, rc.get("buffer_pos_m", 2.0),
                                     "Agranda cada polígono antes de unirlos: las manchas a menos del doble de esta "
                                     "distancia quedan en una misma zona.")
        self.s_buf_neg = FloatSlider("Buffer negativo (m)", 0.0, 20.0, 0.5, rc.get("buffer_neg_m", 1.0),
                                     "Achica las zonas ya unidas. Igual al positivo: solo une y cierra huecos. "
                                     "Menor: deja un margen de seguridad alrededor de la maleza.")
        self.c_bg = QCheckBox("Agregar fondo al resto del lote")
        self.c_bg.setToolTip("Sin marcar: la prescripción tiene solo las zonas con maleza. Marcado: además, el resto "
                             "del lote (perímetro cargado o, si no hay, el área del orto) con su propia dosis.")
        for w in (self.s_buf_pos, self.s_buf_neg, self.c_bg):
            bl.addWidget(w)
        self.rx_table = QTableWidget(2, 3)
        self.rx_table.setHorizontalHeaderLabels(["Zona", "rx", "ha"])
        self.rx_table.verticalHeader().hide()
        self.rx_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.rx_spins = {}
        for i, (zona, texto, val) in enumerate([("maleza", "Con maleza", rc.get("rx_maleza", 100.0)),
                                                 ("fondo", "Fondo", rc.get("rx_fondo", 0.0))]):
            item = QTableWidgetItem(texto)
            item.setFlags(Qt.ItemIsEnabled)
            self.rx_table.setItem(i, 0, item)
            spin = QDoubleSpinBox()
            spin.setRange(0, 100000)
            spin.setDecimals(2)
            spin.setValue(val)
            spin.setFrame(False)
            self.rx_table.setCellWidget(i, 1, spin)
            ha = QTableWidgetItem("—")
            ha.setFlags(Qt.ItemIsEnabled)
            self.rx_table.setItem(i, 2, ha)
            self.rx_spins[zona] = spin
        self.rx_table.setFixedHeight(self.rx_table.horizontalHeader().height() + 2 * self.rx_table.rowHeight(0) + 4)
        bl.addWidget(self.rx_table)
        row = QHBoxLayout()
        row.addWidget(QLabel("Unidad:"))
        self.rx_unit = QComboBox()
        self.rx_unit.addItems(list(UNITS))
        self.rx_unit.setCurrentText(rc.get("unidad", "L/ha"))
        row.addWidget(self.rx_unit, 1)
        bl.addLayout(row)
        self.rx_btn = QPushButton("Generar prescripción")
        self.rx_btn.setMinimumHeight(34)
        bl.addWidget(self.rx_btn)
        self.rx_status = label("", "hint")
        bl.addWidget(self.rx_status)
        bl.addWidget(label("Formatos de exportación:"))
        self.rx_formats = {}
        for key, text in FORMATS.items():
            c = QCheckBox(text)
            c.setChecked(key == "shp")
            bl.addWidget(c)
            self.rx_formats[key] = c
        self.rx_export_btn = QPushButton("Exportar prescripción…")
        self.rx_export_btn.setEnabled(False)
        bl.addWidget(self.rx_export_btn)
        lay.addWidget(box)

        self.result_box = box = QGroupBox("5. Resultado")
        bl = QVBoxLayout(box)
        grid = QGridLayout()
        self.cards = {"total": Card("Hectáreas totales"), "weed": Card("Hectáreas con malezas"),
                      "clean": Card("Hectáreas sin malezas"), "pct": Card("Porcentaje del lote a aplicar")}
        for i, c in enumerate(self.cards.values()):
            grid.addWidget(c, i // 2, i % 2)
        bl.addLayout(grid)
        self.status = label("", "hint")
        bl.addWidget(self.status)
        bl.addWidget(label("Rojo: detectado por el modelo · Naranja: verde sin detección", "hint"))
        self.c_boxes = QCheckBox("Mostrar cajas del modelo en el mapa")
        bl.addWidget(self.c_boxes)
        self.export_btn = QPushButton("Exportar resultados (GPKG + GeoJSON + SHP)…")
        bl.addWidget(self.export_btn)
        lay.addWidget(box)
        lay.addStretch(1)

        self.weights_combo.currentIndexChanged.connect(self.on_selection_changed)
        self.ortho_combo.currentIndexChanged.connect(self.on_selection_changed)
        self.open_btn.clicked.connect(self.open_ortho)
        self.perim_btn.clicked.connect(self.load_perimeter)
        self.perim_clear.clicked.connect(self.clear_perimeter)
        self.lots_combo.currentIndexChanged.connect(self.on_lot_changed)
        self.detect_btn.clicked.connect(self.detect)
        for s in (self.s_conf, self.s_exg, self.s_orphan_area, self.s_orphan_peak, self.s_peak, self.s_area,
                  self.s_border):
            s.valueChanged.connect(self._settings_changed)
        for c in (self.c_orphans, self.c_keep):
            c.toggled.connect(self._settings_changed)
        self.apply_btn.clicked.connect(self.apply_settings)
        self.c_orphans.toggled.connect(self.s_orphan_area.setEnabled)
        self.c_orphans.toggled.connect(self.s_orphan_peak.setEnabled)
        self.c_boxes.toggled.connect(lambda *_: self.render_map())
        self.export_btn.clicked.connect(self.export_results)
        for s in (self.s_buf_pos, self.s_buf_neg):
            s.valueChanged.connect(self._rx_changed)
        for spin in self.rx_spins.values():
            spin.valueChanged.connect(self._rx_changed)
        self.c_bg.toggled.connect(self._rx_changed)
        self.c_bg.toggled.connect(self.rx_spins["fondo"].setEnabled)
        self.rx_spins["fondo"].setEnabled(self.c_bg.isChecked())
        self.rx_unit.currentIndexChanged.connect(self._rx_changed)
        self.rx_btn.clicked.connect(self.make_prescription)
        self.rx_export_btn.clicked.connect(self.export_prescription)
        self._set_result_enabled(False)
        return panel

    def _load_lists(self):
        runs_dir = self.cfg["paths"]["runs_dir"]
        weights = available_weights(self.cfg)
        for w in weights:
            self.weights_combo.addItem(os.path.relpath(w, runs_dir), w)
        if not weights:
            self.weights_msg.setText(f"No encontré modelos en {runs_dir}. Copiá tu best.pt a "
                                     "runs/<nombre>/weights/best.pt o esperá a que el entrenamiento guarde el primero.")
            self.weights_msg.show()
        for p in list_orthos(self.cfg):
            self.ortho_combo.addItem(os.path.basename(p), p)
        self.on_selection_changed()

    def _set_result_enabled(self, on):
        self.settings_box.setEnabled(on)
        self.rx_box.setEnabled(on)
        self.result_box.setEnabled(on)
        if not on:
            for c in self.cards.values():
                c.set("—")
            self.status.setText("")

    # ---------- utilidades ----------

    def weights(self):
        return self.weights_combo.currentData()

    def base_ortho(self):
        """Orto elegido, ya en metros: su copia en UTM si venía en grados (None mientras se reproyecta)."""
        selected = self.ortho_combo.currentData()
        return self.utm.get(selected, selected)

    def ortho(self):
        """Orto con el que se trabaja: el recortado al lote si hay perímetro (None mientras se recorta)."""
        base = self.base_ortho()
        if not base or not self.perim:
            return base
        return self.clips.get((base, self.perim["recorte"]))

    def perimeter_geom(self, path):
        """Perímetro en el CRS del orto, o None sin perímetro."""
        if not self.perim:
            return None
        with rasterio.open(path) as src:
            return perimeter_in(self.perim["gdf"], src.crs)

    def run(self, fn, on_done, on_progress=None, on_fail=None):
        task = Task(fn)
        self._tasks.add(task)  # sin referencia, Python lo libera antes de que emita

        def finish(*_):
            self._tasks.discard(task)

        def fail(tb):
            print(tb, file=sys.stderr)
            if on_fail:
                on_fail()
            QMessageBox.critical(self, "Error", tb.strip().splitlines()[-1] + "\n\n" + tb)

        task.done.connect(on_done)
        task.failed.connect(fail)
        task.done.connect(finish)
        task.failed.connect(finish)
        if on_progress:
            task.progress.connect(on_progress)
        task.start()

    # ---------- 1. ortomosaico ----------

    def open_ortho(self):
        start = os.path.join(self.cfg["root"], "data", "orthos")
        path, _ = QFileDialog.getOpenFileName(self, "Abrir ortomosaico", start, "GeoTIFF (*.tif *.tiff)")
        if not path:
            return
        path = os.path.normpath(path)
        idx = self.ortho_combo.findData(path)
        if idx < 0:
            self.ortho_combo.addItem(os.path.basename(path), path)
            idx = self.ortho_combo.count() - 1
        self.ortho_combo.setCurrentIndex(idx)

    def prepare_ortho(self, selected):
        """Primera vez que se elige un orto: si está en grados (las áreas en m² no servirían), lo reproyecta a UTM."""
        info = ortho_info(selected)
        if not info["geografico"] or info["bandas"] < 3:
            self.utm[selected] = selected
            return
        self.utm[selected] = None
        out = utm_path(selected, self.cfg["paths"]["outputs_dir"])
        for w in (self.ortho_combo, self.open_btn):
            w.setEnabled(False)
        for w in self.info_lbls.values():
            w.setText("—")
        self.ortho_msg.hide()
        self.clip_progress.setValue(0)
        self.clip_progress.setFormat("Reproyectando el ortomosaico a UTM… %p%")
        self.clip_progress.show()
        self.view.setHtml("<body style='font-family:sans-serif;color:#777;padding:2em'>"
                          "Reproyectando el ortomosaico a UTM…</body>")

        def job(emit):
            if not is_fresh(out, selected):
                reproject_to_utm(selected, out, progress=emit)
            return out

        def progress(d, t):
            self.clip_progress.setMaximum(t)
            self.clip_progress.setValue(d)

        def end(path):  # si falla se sigue con el original y queda el aviso de que está en grados
            self.utm[selected] = path
            self.clip_progress.hide()
            for w in (self.ortho_combo, self.open_btn):
                w.setEnabled(True)
            self.on_selection_changed()

        self.run(job, end, on_progress=progress, on_fail=lambda: end(selected))

    # ---------- perímetro del lote ----------

    def load_perimeter(self):
        current = self.perim or self.lots
        start = os.path.dirname(current["path"]) if current else os.path.join(self.cfg["root"], "data")
        path, _ = QFileDialog.getOpenFileName(self, "Cargar perímetro del lote", start, PERIMETER_FILTER)
        if not path:
            return
        path = os.path.normpath(path)
        try:
            lots = read_lots(path)
        except Exception as e:  # archivo sin polígonos, sin CRS, formato no soportado
            QMessageBox.warning(self, "Perímetro del lote", f"No pude leer el perímetro:\n{e}")
            return
        self.perim = self.lots = None
        self.perim_clear.setEnabled(True)
        name = os.path.basename(path)
        self.lots_combo.blockSignals(True)
        self.lots_combo.clear()
        if len(lots) == 1:
            self._use_lot(path, lots, 0)
            self.perim_lbl.setText(f"{name} · {lots['ha'].iloc[0]:,.1f} ha")
        else:
            self.lots = {"path": path, "gdf": lots}
            self.perim_lbl.setText(f"{name}: {len(lots)} lotes. Elegí cuál usar.")
            self.lots_combo.addItem("Todo el ortomosaico (sin recortar)")
            for nombre, ha in zip(lots["nombre"], lots["ha"]):
                self.lots_combo.addItem(f"{nombre} · {ha:,.1f} ha")
        self.lots_combo.blockSignals(False)
        self.lots_combo.setVisible(bool(self.lots))
        self.on_selection_changed()
        self.render_map()

    def clear_perimeter(self):
        self.perim = self.lots = None
        self.perim_lbl.setText("Sin perímetro: se usa todo el ortomosaico.")
        self.perim_clear.setEnabled(False)
        self.lots_combo.hide()
        self.on_selection_changed()
        self.render_map()

    def _use_lot(self, path, lots, i):
        gdf = lot_perimeter(path, lots, i)
        self.perim = {"path": path, "gdf": gdf, "recorte": gdf["recorte"].iloc[0]}

    def selected_lot(self):
        """Posición en la capa del lote elegido; -1 con 'Todo el ortomosaico'."""
        return self.lots_combo.currentIndex() - 1

    def on_lot_changed(self, *_):
        self.perim = None
        if self.selected_lot() >= 0:
            self._use_lot(self.lots["path"], self.lots["gdf"], self.selected_lot())
        self.on_selection_changed()
        self.render_map()

    def _drop_clip(self):
        """El recorte no se pudo hacer: vuelve al orto completo; con una capa de varios lotes, se puede elegir otro."""
        if not self.lots:
            self.clear_perimeter()
            return
        self.lots_combo.blockSignals(True)
        self.lots_combo.setCurrentIndex(0)
        self.lots_combo.blockSignals(False)
        self.on_lot_changed()

    def start_clip(self, base):
        key = (base, self.perim["recorte"])
        if self.clipping == key:
            return
        self.clipping = key
        out = clipped_path(base, key[1], self.cfg["paths"]["outputs_dir"])
        gdf, perim_path = self.perim["gdf"], self.perim["path"]
        self.perim_btn.setEnabled(False)
        self.lots_combo.setEnabled(False)
        self.clip_progress.setValue(0)
        self.clip_progress.setFormat("Recortando el ortomosaico al lote… %p%")
        self.clip_progress.show()

        def job(emit):
            try:
                if not is_fresh(out, base, perim_path):
                    clip_ortho(base, gdf, out, progress=emit)
            except ValueError as e:  # lote fuera del orto o sin datos: no es una falla, se avisa
                return None, str(e)
            return out, None

        def progress(d, t):
            self.clip_progress.setMaximum(t)
            self.clip_progress.setValue(d)

        def end():
            self.clipping = None
            self.clip_progress.hide()
            self.perim_btn.setEnabled(True)
            self.lots_combo.setEnabled(True)

        def done(res):
            out, problem = res
            end()
            if problem:
                self._drop_clip()
                QMessageBox.warning(self, "Perímetro del lote", f"{gdf['nombre'].iloc[0]} {problem}")
                return
            self.clips[key] = out
            self.on_selection_changed()

        def fail():
            end()
            self._drop_clip()

        self.run(job, done, on_progress=progress, on_fail=fail)

    def on_selection_changed(self, *_):
        selected = self.ortho_combo.currentData()
        if selected and selected not in self.utm:
            self.prepare_ortho(selected)
        base = self.base_ortho()
        if base and self.perim and (base, self.perim["recorte"]) not in self.clips:
            self.start_clip(base)
        path = self.ortho()
        if self.det and (self.det["ortho"] != path or self.det["weights"] != self.weights()):
            self.det = self.final = self.polys = None
            self.clear_prescription()  # era de los resultados anteriores: no debe quedar dibujada en el mapa
            self._set_result_enabled(False)
            self.render_map()
        self.detect_btn.setText("Volver a detectar" if self.det else "Detectar malezas")
        if not path:
            self.detect_btn.setEnabled(False)
            return
        if self.info is None or self.info["path"] != path:
            self.show_ortho(path)
        self.detect_btn.setEnabled(bool(self.weights()) and self.info["bandas"] >= 3)

    def show_ortho(self, path):
        info = ortho_info(path)
        self.info = dict(info, path=path)
        res = f"{info['res'][0] * 100:.1f} cm/px" if not info["geografico"] else f"{info['res'][0]:.2e}°"
        for k, v in {"Tamaño": f"{info['ancho']} x {info['alto']} px", "Resolución": res,
                     "Bandas": f"{info['bandas']} ({info['dtype']})", "CRS": info["crs"]}.items():
            self.info_lbls[k].setText(v)

        train_res = self.cfg["predict"].get("train_resolution_m")
        msg, kind = "", "warn"
        if info["bandas"] < 3:
            msg, kind = "Se necesita un ortomosaico RGB (3+ bandas).", "error"
        elif info["geografico"]:
            msg = ("El CRS es geográfico (grados): las áreas en m² no van a ser correctas. "
                   "Reproyectá a UTM (ej. EPSG:32720) antes.")
        elif train_res and abs(info["res"][0] - train_res) / train_res > 0.3:
            msg = (f"La resolución ({info['res'][0] * 100:.1f} cm/px) difiere bastante de la de entrenamiento "
                   f"({train_res * 100:.1f} cm/px): las plantas se ven de otro tamaño y la detección puede empeorar. "
                   "Ideal: volar a la misma altura o remuestrear.")
        if self.base_ortho() != self.ortho_combo.currentData():  # siempre en metros: la copia o su recorte
            note = f"Estaba en grados: se reproyectó solo a {info['crs']} (el archivo original no se toca)."
            msg, kind = f"{note} {msg}" if msg else note, kind if msg else "hint"
        self.ortho_msg.setObjectName(kind)
        self.ortho_msg.style().polish(self.ortho_msg)  # reaplica el estilo según el objectName
        self.ortho_msg.setText(msg)
        self.ortho_msg.setVisible(bool(msg))

        self.map_name = None  # orto nuevo: encuadrarlo
        if path in self.previews:
            self.render_map()
            return
        self.view.setHtml("<body style='font-family:sans-serif;color:#777;padding:2em'>"
                          "Preparando la vista del ortomosaico…</body>")

        def job(_):
            # embebida como data URI: el visor no carga imágenes file:// desde la página del mapa
            img, bounds = ortho_preview(path)
            return path, "data:image/webp;base64," + base64.b64encode(img).decode(), bounds

        def done(res):
            p, url, bounds = res
            self.previews[p] = (url, bounds)
            if p == self.ortho():
                self.render_map()

        self.run(job, done)

    # ---------- 2. detección ----------

    def detect(self):
        path, weights, cfg = self.ortho(), self.weights(), self.cfg
        perim = self.perimeter_geom(path)
        for w in (self.detect_btn, self.weights_combo, self.ortho_combo, self.lots_combo):
            w.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setFormat("Cargando modelo…")
        self.progress.show()

        def job(emit):
            if weights not in self.models:
                self.models[weights] = load_model(weights)
            gdf, n_raw = detect_ortho(self.models[weights], path, cfg, conf=cfg["predict"]["review_conf_min"],
                                      progress=emit)
            return {"ortho": path, "weights": weights, "raw": gdf, "n_raw": n_raw, "perimeter": perim}

        def progress(d, t):
            self.progress.setMaximum(t)
            self.progress.setValue(d)
            self.progress.setFormat(f"Detectando… tile {d}/{t}")

        def end():
            self.progress.hide()
            for w in (self.weights_combo, self.ortho_combo, self.lots_combo):
                w.setEnabled(True)
            self.on_selection_changed()

        def done(det):
            self.det = det
            self._set_result_enabled(True)
            end()
            self.recompute_polys()

        self.run(job, done, on_progress=progress, on_fail=end)

    # ---------- 3. polígonos ----------

    def _settings_changed(self, *_):
        if self.det:
            self.apply_btn.setEnabled(True)
            self.apply_btn.setText("Aplicar ajustes (hay cambios)")
            self.apply_btn.setDefault(True)

    def apply_settings(self):
        self.recompute_polys()

    def params(self):
        pc = dict(self.cfg["polygonize"], exg_threshold=self.s_exg.value(),
                  green_orphan_min_m2=self.s_orphan_area.value(), green_min_peak=self.s_peak.value(),
                  green_orphan_min_peak=self.s_orphan_peak.value(), border_exclude_m=self.s_border.value(),
                  green_min_area_m2=self.s_area.value())
        return {"conf": self.s_conf.value(), "pc": pc, "orphans": self.c_orphans.isChecked(),
                "keep": self.c_keep.isChecked()}

    def green(self, path, pc):
        key = (path, pc["exg_threshold"], pc["green_resolution_m"], pc["green_close_m"], pc.get("green_speck_m2"),
               pc.get("green_peak_window_m"))
        if key not in self.green_cache:
            self.green_cache[key] = green_components(path, dict(self.cfg, polygonize=pc))
            while len(self.green_cache) > 2:
                self.green_cache.popitem(last=False)
        self.green_cache.move_to_end(key)
        return self.green_cache[key]

    def recompute_polys(self):
        if not self.det:
            return
        if self.poly_busy:  # al terminar el cálculo en curso se vuelve a calcular con los ajustes nuevos
            self.poly_pending = True
            return
        self.poly_busy = True
        self.apply_btn.setEnabled(False)  # se reactiva al mover un ajuste (o si el cálculo falla)
        self.apply_btn.setText("Aplicar ajustes")
        self.apply_btn.setDefault(False)
        det, p, cfg = self.det, self.params(), self.cfg
        self.status.setText("Calculando polígonos… (la primera vez se calcula la máscara de verde)")

        def job(_):
            final = det["raw"][det["raw"]["confidence"] >= p["conf"]].reset_index(drop=True)
            if det.get("perimeter") is not None:  # el orto ya está recortado; esto deja el borde exacto del lote
                final = clip_to_perimeter(final, det["perimeter"])
            comps = self.green(det["ortho"], p["pc"])
            polys = polygonize_green(final, det["ortho"], dict(cfg, polygonize=p["pc"]), components=comps,
                                     include_orphans=p["orphans"], keep_no_green=p["keep"])
            if p["pc"]["border_exclude_m"] > 0:
                if det["ortho"] not in self.footprints:
                    self.footprints[det["ortho"]] = ortho_footprint(det["ortho"])
                polys = clip_border(polys, self.footprints[det["ortho"]], p["pc"]["border_exclude_m"])
            if det.get("perimeter") is not None:
                polys = clip_to_perimeter(polys, det["perimeter"])
            return det, final, polys, p["conf"], self.lot_boundary(det).area

        def done(res):
            self.poly_busy = False
            d, final, polys, conf, lot_m2 = res
            if d is self.det:
                self.final, self.polys, self.applied_conf, self.lot_m2 = final, polys, conf, lot_m2
                self.clear_prescription()  # se generó con los polígonos anteriores
                self.show_metrics()
                self.render_map()
            if self.poly_pending or d is not self.det:
                self.poly_pending = False
                self.recompute_polys()

        def fail():
            self.poly_busy = self.poly_pending = False
            self.status.setText("Error al calcular los polígonos.")
            self.apply_btn.setEnabled(True)

        self.run(job, done, on_fail=fail)

    def show_metrics(self):
        polys, n = self.polys, len(self.polys)
        # con maleza: las zonas de la prescripción (con buffers) si ya se generó; si no, los polígonos detectados
        if self.rx is not None:
            weed_ha = float(self.rx.loc[self.rx["zona"] == "maleza", "ha"].sum())
            origen = "zonas de la prescripción"
        else:
            weed_ha = polys["area_m2"].sum() / 1e4 if n else 0.0
            origen = "polígonos detectados, sin buffers"
        lot_ha = self.lot_m2 / 1e4
        self.cards["total"].set(f"{lot_ha:,.2f} ha")
        self.cards["weed"].set(f"{weed_ha:,.2f} ha")
        self.cards["clean"].set(f"{max(lot_ha - weed_ha, 0):,.2f} ha")
        self.cards["pct"].set(f"{100 * weed_ha / lot_ha:.1f} %" if lot_ha else "—")
        self.cards["total"].setToolTip("Perímetro del lote cargado o, sin perímetro, el área con datos del orto.")
        for k in ("weed", "clean", "pct"):
            self.cards[k].setToolTip(f"Calculado con: {origen}.")
        self.status.setText(f"Hectáreas con malezas según: {origen}.\n{n} polígonos de {len(self.final)} "
                            f"detecciones con confianza ≥ {self.applied_conf:.2f} (de {len(self.det['raw'])} crudas)")
        self.export_btn.setEnabled(bool(n))

    # ---------- 4. prescripción ----------

    def _rx_changed(self, *_):
        if self.rx is not None:
            self.rx_btn.setText("Generar prescripción (hay cambios)")

    def clear_prescription(self):
        self.rx = self.rx_boundary = None
        self.rx_btn.setText("Generar prescripción")
        self.rx_status.setText("")
        self.rx_export_btn.setEnabled(False)
        for i in range(2):
            self.rx_table.item(i, 2).setText("—")

    def lot_boundary(self, det):
        """Límite del lote: el perímetro cargado o, sin perímetro, el área con datos del orto."""
        if det.get("perimeter") is not None:
            return det["perimeter"]
        if det["ortho"] not in self.footprints:
            self.footprints[det["ortho"]] = ortho_footprint(det["ortho"])
        return self.footprints[det["ortho"]]

    def make_prescription(self):
        if self.polys is None:
            return
        det, polys = self.det, self.polys
        args = dict(buf_pos=self.s_buf_pos.value(), buf_neg=self.s_buf_neg.value(),
                    rx_weed=self.rx_spins["maleza"].value(), rx_bg=self.rx_spins["fondo"].value(),
                    background=self.c_bg.isChecked(), unit=self.rx_unit.currentText())
        self.rx_btn.setEnabled(False)
        self.rx_status.setText("Generando prescripción…")

        def job(_):
            boundary = self.lot_boundary(det)
            return polys, boundary, build_prescription(polys, boundary, **args)

        def done(res):
            self.rx_btn.setEnabled(True)
            p, boundary, rx = res
            if p is not self.polys:  # cambiaron los polígonos mientras se calculaba
                return
            self.rx, self.rx_boundary = rx, boundary
            self.rx_btn.setText("Generar prescripción")
            s, u = summary(rx, boundary.area), args["unit"]
            self.rx_table.item(0, 2).setText(f"{s['maleza_ha']:.2f}")
            self.rx_table.item(1, 2).setText(f"{s['fondo_ha']:.2f}" if args["background"] else "—")
            prod_u = u.split("/")[0]
            self.rx_status.setText(f"Producto total: {s['maleza_producto'] + s['fondo_producto']:,.1f} {prod_u}")
            self.rx_export_btn.setEnabled(len(rx) > 0)
            self.show_metrics()
            self.render_map()

        self.run(job, done, on_fail=lambda: (self.rx_btn.setEnabled(True), self.rx_status.setText("")))

    def export_prescription(self):
        formats = [k for k, c in self.rx_formats.items() if c.isChecked()]
        if not formats:
            QMessageBox.information(self, "Exportar prescripción", "Marcá al menos un formato.")
            return
        name = safe_name(self.perim["gdf"]["nombre"].iloc[0] if self.perim else self.base_ortho())
        out_dir = os.path.join(self.cfg["paths"]["outputs_dir"], Path(self.base_ortho()).stem, "prescripcion")
        os.makedirs(out_dir, exist_ok=True)
        out_dir = QFileDialog.getExistingDirectory(self, "Carpeta donde guardar la prescripción", out_dir)
        if not out_dir:
            return
        cell = self.cfg.get("prescripcion", {}).get("celda_m", 1.0)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            paths = export_prescription(self.rx, self.rx_boundary, out_dir, name, formats, cell,
                                        self.rx_unit.currentText())
        finally:
            QApplication.restoreOverrideCursor()
        lines = "\n".join(f"• {FORMATS[k].split(' (')[0]}: {os.path.relpath(p, out_dir)}" for k, p in paths.items())
        box = QMessageBox(QMessageBox.Information, "Exportar prescripción", f"Guardado en {out_dir}:\n\n{lines}",
                          parent=self)
        open_btn = box.addButton("Abrir carpeta", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if box.clickedButton() is open_btn:
            os.startfile(out_dir)

    # ---------- mapa ----------

    def render_map(self):
        """Redibuja el mapa conservando el encuadre actual (salvo con un orto recién elegido)."""
        path = self.ortho()
        if path not in self.previews:
            return
        url, bounds = self.previews[path]

        def draw(view_json):
            try:
                view = json.loads(view_json) if view_json else None
            except ValueError:
                view = None
            lots = None
            if self.lots:  # toda la capa, con su nombre y resaltado el lote elegido
                lots = self.lots["gdf"].assign(elegido=[i == self.selected_lot() for i in range(len(self.lots["gdf"]))])
            m = build_map(url, bounds, boxes=self.final, polys=self.polys, view=view,
                          boxes_visible=self.c_boxes.isChecked(), tiles=self.tiles.register(path),
                          perimeter=self.perim["gdf"] if self.perim and not self.lots else None, rx=self.rx, lots=lots)
            self.map_name = m.get_name()
            self.map_n += 1
            # se alterna el nombre del archivo para que el visor lo cargue de nuevo
            out = os.path.join(self.tmp.name, f"mapa_{self.map_n % 2}.html")
            m.save(out)
            self.view.load(QUrl.fromLocalFile(out))

        if self.map_name:
            js = (f"(function(){{try{{var c={self.map_name}.getCenter();"
                  f"return JSON.stringify([c.lat, c.lng, {self.map_name}.getZoom()]);}}catch(e){{return '';}}}})()")
            self.view.page().runJavaScript(js, 0, draw)
        else:
            draw(None)

    # ---------- exportar ----------

    def export_results(self):
        out_dir = os.path.join(self.cfg["paths"]["outputs_dir"], Path(self.det["ortho"]).stem)
        os.makedirs(out_dir, exist_ok=True)
        out_dir = QFileDialog.getExistingDirectory(self, "Carpeta donde guardar los resultados", out_dir)
        if not out_dir:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            export(self.final.drop(columns=["px_x1", "px_y1", "px_x2", "px_y2", "truncated"], errors="ignore"),
                   self.polys, out_dir)
        finally:
            QApplication.restoreOverrideCursor()
        box = QMessageBox(QMessageBox.Information, "Exportar", f"Polígonos y cajas del modelo guardados en:\n{out_dir}",
                          parent=self)
        open_btn = box.addButton("Abrir carpeta", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if box.clickedButton() is open_btn:
            os.startfile(out_dir)

    def closeEvent(self, event):
        self.tiles.close()
        self.tmp.cleanup()
        super().closeEvent(event)


def main(cfg, config_path):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Detección de malezas")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)

    def excepthook(exc_type, exc, tb):  # errores en la UI: avisar en vez de cerrar en silencio
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        print(text, file=sys.stderr)
        QMessageBox.critical(None, "Error", text)

    sys.excepthook = excepthook
    win = MainWindow(cfg, config_path)
    win.showMaximized()
    return app.exec()
