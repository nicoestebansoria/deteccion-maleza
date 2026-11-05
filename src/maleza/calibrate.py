"""Calibración de los ajustes de la app contra las etiquetas a mano (data/labels).

Mide, dentro de los tiles etiquetados, a nivel mancha verde (lo que después se une con buffer/disolve):
- recall: fracción de cajas etiquetadas que tocan alguna mancha del resultado;
- precisión (por área): de la superficie marcada dentro de los tiles evaluados, qué fracción cae sobre cajas.
  Por área y no por cantidad de manchas: contar manchas premia unir todo en pocos manchones gigantes;
- manchas falsas: manchas del resultado que no tocan ninguna caja (cada una se agranda con el buffer).
  Ojo: las etiquetas casi no incluyen plantas chicas, así que muchas de estas son maleza sin etiquetar.
- con respuestas de la revisión rápida (outputs/<orto>/revision_manchas.json): prec_revisada = fracción de
  manchas sin etiqueta que son maleza según las respuestas, y falsas_est / reales_est = cantidad estimada
  en todo el orto (cada respuesta pesa por cuántas manchas de su nivel de verde representa).

El modelo vio las cajas de los tiles de entrenamiento: su efecto (confianza mínima) solo se mide sin sesgo
en los tiles de validación (recall_val). El índice de verde no se entrena, así que el resto vale en todos.
La máscara de verde se calcula una vez por combinación de (umbral ExG, cierre); los demás ajustes son
filtros sobre las manchas y se evalúan en milisegundos.
"""
import hashlib
import itertools
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from .common import load_tiles_metadata, load_yolo_labels, yolo_to_px
from .dataset import labeled_pairs
from .polygonize import green_components
from .predict import detect_ortho, load_model

GRID = {
    "exg_threshold": [0.03, 0.05, 0.07],
    "green_close_m": [0.2, 0.3, 0.5],
    "conf": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6],
    "green_min_peak": [0, 10, 15, 20, 25, 30],
    "green_min_area_m2": [0.02, 0.05, 0.1, 0.2, 0.4],
    "green_orphan_min_m2": [None, 0.25, 0.5, 1.0, 2.0],  # None = sin manchas sin detección
    "green_orphan_min_peak": [20, 25, 30, 35, 40, 50],
}
TOUCH_TOL_M = 0.2  # una mancha "toca" una caja si pasa a menos de esto (bordes de etiquetado imprecisos)


def labeled_boxes(cfg, ortho_path, ortho_key):
    """Cajas etiquetadas en píxeles del orto (sin duplicados por solape de tiles), ventanas de los tiles
    y si cada caja/tile es de validación."""
    p = cfg["paths"]
    meta = load_tiles_metadata(p["tiles_dir"])
    val_dir = os.path.join(cfg["root"], "data", "dataset", "images", "val")
    val_names = set(os.listdir(val_dir)) if os.path.isdir(val_dir) else set()
    boxes, windows = [], []
    for tile_path, label_path in labeled_pairs(p["tiles_dir"], p["labels_dir"]):
        m = meta.get(os.path.basename(tile_path))
        if not m or m["ortho"] != ortho_key:
            continue
        is_val = os.path.basename(tile_path) in val_names
        l, t, w, h = m["window_left"], m["window_top"], m["window_width"], m["window_height"]
        windows.append((l, t, w, h, is_val))
        for b in load_yolo_labels(label_path):
            x1, y1, x2, y2 = yolo_to_px(b, w, h)
            boxes.append((l + x1, t + y1, l + x2, t + y2, is_val))
    boxes = np.array(boxes, dtype=np.float64).reshape(-1, 5)
    # dedup: el mismo arbusto etiquetado en dos tiles solapados (entero en uno, cortado en el otro)
    order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    keep = []
    for i in order:
        x1, y1, x2, y2 = boxes[i, :4]
        dup = False
        for j in keep:
            ix = max(0, min(x2, boxes[j, 2]) - max(x1, boxes[j, 0]))
            iy = max(0, min(y2, boxes[j, 3]) - max(y1, boxes[j, 1]))
            if ix * iy > 0.7 * (x2 - x1) * (y2 - y1):  # casi toda la caja más chica dentro de otra
                dup = True
                break
        if not dup:
            keep.append(i)
    return boxes[sorted(keep)], windows


def cached_detections(cfg, ortho_path, weights):
    """Detecciones crudas (conf >= review_conf_min); se guardan para no correr el modelo en cada calibración."""
    tag = hashlib.md5(f"{weights}|{os.path.getmtime(weights)}".encode()).hexdigest()[:8]
    out = os.path.join(cfg["paths"]["outputs_dir"], Path(ortho_path).stem, f"detecciones_crudas_{tag}.gpkg")
    if os.path.exists(out):
        return gpd.read_file(out)
    print("Corriendo el modelo sobre el orto (una sola vez)…")
    raw, _ = detect_ortho(load_model(weights), ortho_path, cfg, conf=cfg["predict"]["review_conf_min"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    raw[["confidence", "geometry"]].to_file(out, driver="GPKG")
    return raw


def load_answers(cfg, ortho_path):
    """[(x, y, es_maleza, peso)] de la revisión rápida; peso = manchas de su estrato / respuestas del estrato."""
    path = os.path.join(cfg["paths"]["outputs_dir"], Path(ortho_path).stem, "revision_manchas.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        review = json.load(f)
    ans = [m for m in review["muestra"] if m["resp"] in ("si", "no")]
    n_k = {}
    for m in ans:
        n_k[m["estrato"]] = n_k.get(m["estrato"], 0) + 1
    return [(m["x"], m["y"], m["resp"] == "si", review["estratos"][str(m["estrato"])] / n_k[m["estrato"]])
            for m in ans]


def _blob_tables(comps, boxes, windows, dets, ortho_shape, answers=()):
    """Datos por mancha que no dependen de los filtros: área, pico, máx. confianza de detección que la toca,
    si toca alguna caja, si está en zona evaluada; y pares (caja, mancha) para el recall."""
    labels, transform, areas, _, peak = comps
    n = len(areas)
    f = ortho_shape[1] / labels.shape[1]  # factor orto -> máscara (1 a resolución original)
    px = abs(transform.a)
    tol = int(round(TOUCH_TOL_M / px))
    H, W = labels.shape

    def sl(x1, y1, x2, y2, pad=0):
        return (slice(max(0, int(y1 / f) - pad), min(H, int(np.ceil(y2 / f)) + pad)),
                slice(max(0, int(x1 / f) - pad), min(W, int(np.ceil(x2 / f)) + pad)))

    eval_mask = np.zeros(labels.shape, bool)
    for l, t, w, h, _ in windows:
        eval_mask[sl(l, t, l + w, t + h)] = True
    box_mask = np.zeros(labels.shape, bool)
    box_idx, box_lbl = [], []
    for i, (x1, y1, x2, y2, _) in enumerate(boxes):
        s_ = sl(x1, y1, x2, y2, tol)
        box_mask[s_] = True
        found = np.unique(labels[s_])
        found = found[found > 0]
        box_idx.append(np.full(len(found), i))
        box_lbl.append(found)
    npx = np.bincount(labels.ravel(), minlength=n)
    eval_px = np.bincount(labels[eval_mask], minlength=n)   # píxeles de cada mancha dentro de los tiles
    box_px = np.bincount(labels[box_mask & eval_mask], minlength=n)  # y dentro de cajas
    del eval_mask, box_mask
    in_eval = eval_px >= 0.5 * np.maximum(npx, 1)
    touches = box_px > 0

    inv = ~transform
    maxconf = np.zeros(n, np.float32)
    for geom, conf in zip(dets.geometry, dets["confidence"].to_numpy()):
        minx, miny, maxx, maxy = geom.bounds
        c1, r1 = inv * (minx, maxy)
        c2, r2 = inv * (maxx, miny)
        found = np.unique(labels[max(0, int(r1)):min(H, int(np.ceil(r2))), max(0, int(c1)):min(W, int(np.ceil(c2)))])
        np.maximum.at(maxconf, found, conf)
    maxconf[0] = 0
    yes_w, no_w = np.zeros(n), np.zeros(n)
    for x, y, es, w in answers:
        c, r = inv * (x, y)
        if 0 <= r < H and 0 <= c < W:
            (yes_w if es else no_w)[labels[int(r), int(c)]] += w
    yes_w[0] = no_w[0] = 0
    return {"yes_w": yes_w, "no_w": no_w,"areas": areas, "peak": peak, "maxconf": maxconf, "touches": touches, "in_eval": in_eval,
            "eval_m2": eval_px * px * px, "box_m2": box_px * px * px,
            "box_idx": np.concatenate(box_idx) if box_idx else np.array([], int),
            "box_lbl": np.concatenate(box_lbl) if box_lbl else np.array([], int)}


def _evaluate(t, boxes, conf, peak, min_area, orphan_min, orphan_peak=0.0):
    valid = (t["areas"] >= min_area) & (t["peak"] >= peak)
    valid[0] = False
    sel = valid & (t["maxconf"] >= conf)
    if orphan_min is not None:
        sel |= valid & (t["areas"] >= orphan_min) & (t["peak"] >= orphan_peak)
    hit = np.bincount(t["box_idx"], weights=sel[t["box_lbl"]], minlength=len(boxes)) > 0
    is_val = boxes[:, 4] > 0
    ev = sel & t["in_eval"]
    m2_eval, m2_box = t["eval_m2"][sel].sum(), t["box_m2"][sel].sum()
    recall, prec = hit.mean(), (m2_box / m2_eval if m2_eval else 0.0)
    return {"recall": recall, "recall_val": hit[is_val].mean() if is_val.any() else np.nan,
            "precision": prec, "f1": 2 * prec * recall / (prec + recall) if prec + recall else 0.0,
            "manchas": int(ev.sum()), "manchas_falsas": int((ev & ~t["touches"]).sum()),
            "m2_sobre_cajas": float(m2_box), "m2_fuera": float(m2_eval - m2_box),
            **({"prec_revisada": t["yes_w"][sel].sum() / max((t["yes_w"] + t["no_w"])[sel].sum(), 1e-9),
                "reales_est": float(t["yes_w"][sel].sum()), "falsas_est": float(t["no_w"][sel].sum())}
               if (t["yes_w"].any() or t["no_w"].any()) else {})}


def calibrate(cfg, ortho_path, ortho_key, weights, grid=None):
    grid = {**GRID, **(grid or {})}
    boxes, windows = labeled_boxes(cfg, ortho_path, ortho_key)
    print(f"Etiquetas: {len(boxes)} cajas en {len(windows)} tiles "
          f"({int(boxes[:, 4].sum())} cajas en {sum(w[4] for w in windows)} tiles de validación)")
    dets = cached_detections(cfg, ortho_path, weights)
    answers = load_answers(cfg, ortho_path)
    if answers:
        print(f"Revisión rápida: {len(answers)} respuestas ({sum(a[2] for a in answers)} sí)")
    with rasterio.open(ortho_path) as src:
        shape = (src.height, src.width)

    rows = []
    for exg, close in itertools.product(grid["exg_threshold"], grid["green_close_m"]):
        pc = dict(cfg["polygonize"], exg_threshold=exg, green_close_m=close)
        comps = green_components(ortho_path, dict(cfg, polygonize=pc))
        t = _blob_tables(comps, boxes, windows, dets, shape, answers)
        del comps
        for conf, peak, area, orph, opk in itertools.product(
                grid["conf"], grid["green_min_peak"], grid["green_min_area_m2"], grid["green_orphan_min_m2"],
                grid["green_orphan_min_peak"]):
            if orph is None and opk != grid["green_orphan_min_peak"][0]:
                continue  # sin manchas sin detección el umbral de estas no importa
            rows.append({"exg_threshold": exg, "green_close_m": close, "conf": conf, "green_min_peak": peak,
                         "green_min_area_m2": area, "green_orphan_min_m2": orph, "green_orphan_min_peak": opk,
                         **_evaluate(t, boxes, conf, peak, area, orph, opk)})
        print(f"  ExG {exg} · cierre {close} m: listo")
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    out = os.path.join(cfg["paths"]["outputs_dir"], Path(ortho_path).stem, "calibracion.csv")
    df.to_csv(out, index=False)
    print(f"Tabla completa: {out}")
    return df
