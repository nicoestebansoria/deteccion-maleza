"""Revisión rápida de manchas verdes (TEMPORAL): el usuario responde sí/no sobre una muestra de manchas y
`maleza calibrar` usa las respuestas para estimar la precisión real de cada combinación de ajustes
(las etiquetas a mano casi no incluyen plantas chicas, así que no sirven para medir falsos positivos).

Para sacarla: borrar este archivo y la pestaña en app.py (buscar "revision_rapida").
Las respuestas quedan en outputs/<orto>/revision_manchas.json.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

# estratos por verde pico (2G-R-B): la muestra se reparte parejo desde el verde tenue (dudoso) al intenso
PEAK_BINS = [10, 20, 30, 45, np.inf]
MIN_AREA_M2 = 0.05
VIEW_PX = 440  # lado de cada imagen en la ventana de revisión


def answers_path(cfg, ortho_path):
    return os.path.join(cfg["paths"]["outputs_dir"], Path(ortho_path).stem, "revision_manchas.json")


def load_review(cfg, ortho_path):
    path = answers_path(cfg, ortho_path)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_review(cfg, ortho_path, review):
    path = answers_path(cfg, ortho_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(review, f)
    os.replace(tmp, path)


def build_sample(cfg, ortho_path, comps, pc, n_per_bin=50, seed=None):
    """Muestra estratificada de manchas sin etiqueta. Guarda un punto interior y el contorno de cada una."""
    from .calibrate import labeled_boxes
    labels, transform, areas, _, peak = comps
    H, W = labels.shape
    cand = (areas >= MIN_AREA_M2) & (peak >= PEAK_BINS[0])
    cand[0] = False

    # las que tocan una etiqueta ya se saben maleza: no se preguntan
    key = next((o["prefix"] or "o1" for o in cfg["orthos"] if os.path.normcase(o["path"]) == os.path.normcase(ortho_path)),
               None)
    if key:
        boxes, _ = labeled_boxes(cfg, ortho_path, key)
        with rasterio.open(ortho_path) as src:
            f = src.width / W
        for x1, y1, x2, y2, _ in boxes:
            found = np.unique(labels[max(0, int(y1 / f)):min(H, int(np.ceil(y2 / f))),
                                     max(0, int(x1 / f)):min(W, int(np.ceil(x2 / f)))])
            cand[found] = False

    rng = np.random.default_rng(seed)
    strata = np.digitize(peak, PEAK_BINS) - 1
    picks, counts = [], {}
    for k in range(len(PEAK_BINS) - 1):
        ids = np.flatnonzero(cand & (strata == k))
        counts[str(k)] = int(len(ids))
        picks += [(int(i), k) for i in rng.choice(ids, min(n_per_bin, len(ids)), replace=False)]
    rng.shuffle(picks)

    ids = np.array([i for i, _ in picks])
    ys, xs = np.nonzero(np.isin(labels, ids))
    lab = labels[ys, xs]
    order = np.argsort(lab, kind="stable")
    ys, xs, lab = ys[order], xs[order], lab[order]
    starts = np.searchsorted(lab, ids)
    ends = np.searchsorted(lab, ids, side="right")
    items = []
    for (i, k), s, e in zip(picks, starts, ends):
        py, px_ = ys[s:e], xs[s:e]
        # punto interior: el píxel de la mancha más cercano a su centroide (el centroide puede caer afuera)
        j = np.argmin((py - py.mean()) ** 2 + (px_ - px_.mean()) ** 2)
        x, y = transform * (px_[j] + 0.5, py[j] + 0.5)
        r0, r1, c0, c1 = py.min(), py.max() + 1, px_.min(), px_.max() + 1
        mask = (labels[r0:r1, c0:c1] == i).astype(np.uint8)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rings = [[list(transform * (c0 + p[0][0], r0 + p[0][1])) for p in c] for c in cs]
        items.append({"x": x, "y": y, "area": round(float(areas[i]), 3), "peak": int(round(float(peak[i]))),
                      "estrato": k, "contorno": rings, "resp": None})
    return {"orto": ortho_path, "exg_threshold": pc["exg_threshold"], "green_close_m": pc["green_close_m"],
            "estratos": counts, "muestra": items}


def summary(review):
    resp = [m["resp"] for m in review["muestra"]]
    return {"total": len(resp), "si": resp.count("si"), "no": resp.count("no"), "nose": resp.count("nose"),
            "pendientes": resp.count(None)}


class ReviewDialog(QDialog):
    """Ventana de revisión: la mancha sin marcar y marcada, lado a lado. S = sí, N = no, Espacio = no sé."""

    def __init__(self, parent, cfg, ortho_path, review, on_change):
        super().__init__(parent)
        self.cfg, self.path, self.review, self.on_change = cfg, ortho_path, review, on_change
        self.src = rasterio.open(ortho_path)
        self.setWindowTitle("Revisión rápida de manchas")
        self.history = []

        self.info = QLabel()
        self.info.setStyleSheet("font-size: 11pt;")
        self.view_m = QComboBox()
        for m in (5, 10, 20):
            self.view_m.addItem(f"Ventana de {m} m", m)
        self.view_m.setCurrentIndex(0)  # 5 m: las manchas chicas se ven bien
        self.view_m.setFocusPolicy(Qt.NoFocus)
        head = QHBoxLayout()
        head.addWidget(self.info, 1)
        head.addWidget(self.view_m)

        self.img_raw, self.img_mark = QLabel(), QLabel()
        for w in (self.img_raw, self.img_mark):
            w.setFixedSize(VIEW_PX, VIEW_PX)
            w.setAlignment(Qt.AlignCenter)
        imgs = QHBoxLayout()
        imgs.addWidget(self.img_raw)
        imgs.addWidget(self.img_mark)

        buttons = QHBoxLayout()
        self.btns = {}
        for resp, text, key in (("si", "Sí, es maleza  [S]", "S"), ("no", "No  [N]", "N"),
                                ("nose", "No sé  [Espacio]", "Space")):
            b = QPushButton(text)
            b.setMinimumHeight(40)
            b.setFocusPolicy(Qt.NoFocus)  # si no, Espacio "aprieta" el botón enfocado además del atajo
            b.clicked.connect(lambda _=False, r=resp: self.answer(r))
            QShortcut(QKeySequence(key), self, activated=lambda r=resp: self.answer(r))
            buttons.addWidget(b)
        back = QPushButton("← Anterior  [Retroceso]")
        back.setMinimumHeight(40)
        back.setFocusPolicy(Qt.NoFocus)
        back.clicked.connect(self.back)
        QShortcut(QKeySequence("Backspace"), self, activated=self.back)
        buttons.addWidget(back)

        hint = QLabel("La mancha está marcada en violeta a la derecha. Respondé si lo que hay adentro es una planta "
                      "(maleza) o no (rastrojo, suelo, sombra). En la duda, «No sé».")
        hint.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.addLayout(head)
        lay.addLayout(imgs)
        lay.addWidget(hint)
        lay.addLayout(buttons)
        self.view_m.currentIndexChanged.connect(lambda *_: self.show_item())
        self.idx = self._next_pending(0)
        self.show_item()

    def _next_pending(self, start):
        items = self.review["muestra"]
        for i in list(range(start, len(items))) + list(range(0, start)):
            if items[i]["resp"] is None:
                return i
        return None

    def show_item(self):
        s = summary(self.review)
        if self.idx is None:
            self.info.setText(f"<b>¡Listo!</b> {s['si']} sí · {s['no']} no · {s['nose']} no sé. "
                              "Ya podés cerrar esta ventana.")
            for w in (self.img_raw, self.img_mark):
                w.clear()
            return
        m = self.review["muestra"][self.idx]
        done = s["total"] - s["pendientes"]
        self.info.setText(f"<b>{done + 1} / {s['total']}</b> · área {m['area']} m² · verde pico {m['peak']} "
                          f"· respondidas: {s['si']} sí, {s['no']} no, {s['nose']} no sé")
        side = self.view_m.currentData()
        res = abs(self.src.transform.a)
        n = int(round(side / res))
        row, col = self.src.index(m["x"], m["y"])
        win = Window(col - n // 2, row - n // 2, n, n)
        rgb = np.moveaxis(self.src.read([1, 2, 3], window=win, boundless=True), 0, -1)
        if rgb.dtype != np.uint8:
            lo, hi = np.percentile(rgb[rgb > 0], [2, 98]) if (rgb > 0).any() else (0, 1)
            rgb = (np.clip((rgb - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
        scale = VIEW_PX / n
        raw = cv2.resize(np.ascontiguousarray(rgb), (VIEW_PX, VIEW_PX), interpolation=cv2.INTER_LINEAR)
        mark = raw.copy()
        inv = ~self.src.window_transform(win)
        for ring in m["contorno"]:
            pts = np.array([[(inv * tuple(p))[0] * scale, (inv * tuple(p))[1] * scale] for p in ring], np.int32)
            cv2.polylines(mark, [pts], True, (255, 0, 255), 2)
        for w, a in ((self.img_raw, raw), (self.img_mark, mark)):
            a = np.ascontiguousarray(a)
            w.setPixmap(QPixmap.fromImage(QImage(a.data, VIEW_PX, VIEW_PX, 3 * VIEW_PX, QImage.Format_RGB888).copy()))

    def answer(self, resp):
        if self.idx is None:
            return
        self.review["muestra"][self.idx]["resp"] = resp
        save_review(self.cfg, self.path, self.review)
        self.history.append(self.idx)
        self.idx = self._next_pending(self.idx + 1)
        self.on_change()
        self.show_item()

    def back(self):
        if not self.history:
            return
        self.idx = self.history.pop()
        self.review["muestra"][self.idx]["resp"] = None
        save_review(self.cfg, self.path, self.review)
        self.on_change()
        self.show_item()

    def closeEvent(self, event):
        self.src.close()
        super().closeEvent(event)


class ReviewTab(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win, self.cfg = win, win.cfg
        lay = QVBoxLayout(self)
        intro = QLabel(
            "<b>Sección temporal.</b> Muestra manchas verdes al azar (sin etiqueta a mano) para responder si son "
            "maleza o no. Con unas 150 respuestas, <code>maleza calibrar</code> estima cuántas manchas falsas deja "
            "cada combinación de ajustes y se puede recomendar una configuración.<br><br>"
            "La muestra se arma con el umbral de verde y el cierre actuales de la pestaña Detección, e incluye "
            "manchas dudosas a propósito (verde tenue), así que no importa si muchas son «No».")
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        lay.addWidget(intro)
        row = QHBoxLayout()
        row.addWidget(QLabel("Manchas por nivel de verde (4 niveles):"))
        self.n_per = QSpinBox()
        self.n_per.setRange(10, 200)
        self.n_per.setValue(50)
        row.addWidget(self.n_per)
        lay.addLayout(row)
        self.gen_btn = QPushButton("Generar muestra nueva")
        self.open_btn = QPushButton("Revisar…")
        self.open_btn.setMinimumHeight(36)
        self.status = QLabel()
        self.status.setWordWrap(True)
        lay.addWidget(self.gen_btn)
        lay.addWidget(self.open_btn)
        lay.addWidget(self.status)
        lay.addStretch(1)
        self.gen_btn.clicked.connect(self.generate)
        self.open_btn.clicked.connect(self.open_review)
        win.ortho_combo.currentIndexChanged.connect(lambda *_: self.refresh())
        self.refresh()

    def refresh(self):
        path = self.win.ortho()
        review = load_review(self.cfg, path) if path else None
        self.open_btn.setEnabled(bool(review))
        if not review:
            self.status.setText("No hay muestra para este ortomosaico.")
            return
        s = summary(review)
        self.status.setText(f"Muestra de {s['total']} manchas: {s['si']} sí · {s['no']} no · {s['nose']} no sé · "
                            f"{s['pendientes']} pendientes.\nGuardado en {answers_path(self.cfg, path)}")

    def generate(self):
        path = self.win.ortho()
        if not path:
            return
        old = load_review(self.cfg, path)
        if old and summary(old)["pendientes"] < summary(old)["total"]:
            if QMessageBox.question(self, "Muestra nueva", "Ya hay respuestas para este ortomosaico. "
                                    "¿Reemplazarlas por una muestra nueva?") != QMessageBox.Yes:
                return
        pc = self.win.params()["pc"]
        n_per = self.n_per.value()
        self.gen_btn.setEnabled(False)
        self.status.setText("Armando la muestra… (calcula la máscara de verde si no está en memoria)")

        def job(_):
            return build_sample(self.cfg, path, self.win.green(path, pc), pc, n_per_bin=n_per)

        def done(review):
            save_review(self.cfg, path, review)
            self.gen_btn.setEnabled(True)
            self.refresh()
            self.open_review()

        self.win.run(job, done, on_fail=lambda: self.gen_btn.setEnabled(True))

    def open_review(self):
        path = self.win.ortho()
        review = load_review(self.cfg, path)
        if review:
            ReviewDialog(self, self.cfg, path, review, self.refresh).exec()
