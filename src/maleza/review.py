"""Revisión manual asistida: ranking de tiles dudosos, correcciones y pool de entrenamiento.

Las correcciones de un tile reemplazan a las detecciones del modelo cuyo centro cae en ese
tile. Si el ortomosaico se marca para entrenamiento, cada tile revisado se guarda como
tile + label YOLO + metadata (georreferencia), igual que los tiles etiquetados en Colab.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

from .common import iter_windows, read_tile, write_yolo_labels
from .predict import boxes_to_gdf, merge_tiles_nms, truncated_flags


def ortho_slug(ortho_path):
    """Identificador del ortomosaico para nombres de tiles/metadata (ej. 'lote_norte_2026')."""
    return re.sub(r"[^a-z0-9]+", "_", Path(ortho_path).stem.lower()).strip("_")[:40]


def window_key(window):
    return f"r{int(window.row_off):05d}_c{int(window.col_off):05d}"


def tile_windows(ortho_path, cfg):
    """Ventanas del recorrido de inferencia con datos válidos -> {key: Window}."""
    t, pr = cfg["tiling"], cfg["predict"]
    out = {}
    with rasterio.open(ortho_path) as src:
        for w in iter_windows(src.width, src.height, t["tile_size"], t["overlap"]):
            mask = src.dataset_mask(window=w, out_shape=(64, 64))
            if (mask == 0).mean() <= pr["nodata_threshold"]:
                out[window_key(w)] = w
    return out


def _centers(gdf):
    return (gdf["px_x1"].values + gdf["px_x2"].values) / 2, (gdf["px_y1"].values + gdf["px_y2"].values) / 2


def _in_window(cx, cy, w):
    return (cx >= w.col_off) & (cx < w.col_off + w.width) & (cy >= w.row_off) & (cy < w.row_off + w.height)


def rank_tiles(gdf, windows, conf_thr, conf_min, margin=0.2):
    """Tiles ordenados por cantidad de detecciones dudosas (conf entre conf_min y conf_thr+margin).

    -> lista de dicts {key, window, n, dudosas}, primero los más dudosos.
    """
    cx, cy = _centers(gdf)
    conf = gdf["confidence"].values
    dudosa = (conf >= conf_min) & (conf < conf_thr + margin)
    firme = conf >= conf_thr
    rows = []
    for key, w in windows.items():
        inside = _in_window(cx, cy, w)
        rows.append({"key": key, "window": w, "n": int((inside & firme).sum()),
                     "dudosas": int((inside & dudosa).sum())})
    rows.sort(key=lambda r: (-r["dudosas"], -r["n"], r["key"]))
    return rows


def tile_image(ortho_path, window, tile_size):
    with rasterio.open(ortho_path) as src:
        tile, _ = read_tile(src, window, tile_size)
    if tile.dtype != np.uint8:
        tile = np.clip(tile / max(np.percentile(tile, 99.5), 1) * 255, 0, 255).astype(np.uint8)
    return tile


def boxes_in_window(gdf, window, conf_min):
    """Cajas con centro en la ventana -> [[x, y, w, h], ...] en coords del tile."""
    cx, cy = _centers(gdf)
    sel = gdf[_in_window(cx, cy, window) & (gdf["confidence"].values >= conf_min)]
    out = []
    for r in sel.itertuples():
        x1, y1 = max(0.0, r.px_x1 - window.col_off), max(0.0, r.px_y1 - window.row_off)
        x2 = min(float(window.width), r.px_x2 - window.col_off)
        y2 = min(float(window.height), r.px_y2 - window.row_off)
        if x2 > x1 and y2 > y1:
            out.append([x1, y1, x2 - x1, y2 - y1])
    return out


def clean_boxes(boxes_xywh, window, min_px=3):
    """Recorta al área válida del tile (sin padding) y descarta cajas degeneradas."""
    out = []
    for x, y, w, h in boxes_xywh:
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(window.width), x + w), min(float(window.height), y + h)
        if x2 - x1 >= min_px and y2 - y1 >= min_px:
            out.append([x1, y1, x2 - x1, y2 - y1])
    return out


def apply_corrections(gdf, corrections, conf_thr, ortho_path, cfg):
    """Detecciones finales: modelo filtrado por conf_thr + correcciones manuales por tile."""
    pr = cfg["predict"]
    base = gdf[gdf["confidence"] >= conf_thr]
    if not corrections:
        return base.reset_index(drop=True)

    with rasterio.open(ortho_path) as src:
        transform, crs, W, H = src.transform, src.crs, src.width, src.height

    cx, cy = _centers(base)
    replaced = np.zeros(len(base), bool)
    boxes, truncated = [], []
    for c in corrections.values():
        w = Window(*c["window"])
        replaced |= _in_window(cx, cy, w)
        if c["boxes"]:
            xyxy = np.array([[x, y, x + bw, y + bh] for x, y, bw, bh in c["boxes"]], float)
            truncated.append(truncated_flags(xyxy, w, W, H))
            xyxy[:, [0, 2]] += w.col_off
            xyxy[:, [1, 3]] += w.row_off
            boxes.append(xyxy)

    kept = base[~replaced]
    if boxes:
        boxes_px = np.concatenate(boxes)
        trunc = np.concatenate(truncated)
        # tiles revisados que se solapan pueden tener la misma planta dos veces
        keep = merge_tiles_nms(boxes_px, np.ones(len(boxes_px)), trunc, pr["dedup_iou"], pr["dedup_ios"])
        manual = boxes_to_gdf(boxes_px[keep], np.ones(len(keep)), trunc[keep], transform, crs, source="manual")
        return gpd_concat([kept, manual], crs)
    return kept.reset_index(drop=True)


def gpd_concat(frames, crs):
    import geopandas as gpd
    import pandas as pd
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=crs)


# --- persistencia de la revisión ---

def revision_path(cfg, ortho_path):
    return os.path.join(cfg["paths"]["outputs_dir"], Path(ortho_path).stem, "revision.json")


def load_revision(cfg, ortho_path):
    p = revision_path(cfg, ortho_path)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"entrenamiento": False, "tiles": {}}


def save_revision(cfg, ortho_path, revision):
    p = revision_path(cfg, ortho_path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(revision, f, indent=1)


# --- pool de entrenamiento ---

def training_tile_name(ortho_path, window):
    return f"tile_{ortho_slug(ortho_path)}_{window_key(window)}"


def save_training_tile(ortho_path, window, boxes_xywh, cfg):
    """Guarda tile JPG + label YOLO + entrada de metadata para el split espacial."""
    p, tile_size = cfg["paths"], cfg["tiling"]["tile_size"]
    os.makedirs(p["tiles_dir"], exist_ok=True)
    os.makedirs(p["labels_dir"], exist_ok=True)
    name = training_tile_name(ortho_path, window)
    Image.fromarray(tile_image(ortho_path, window, tile_size)).save(
        os.path.join(p["tiles_dir"], f"{name}.jpg"), quality=95)
    write_yolo_labels(os.path.join(p["labels_dir"], f"{name}.txt"),
                      [((x + w / 2) / tile_size, (y + h / 2) / tile_size, w / tile_size, h / tile_size)
                       for x, y, w, h in boxes_xywh])

    slug = ortho_slug(ortho_path)
    meta_path = os.path.join(p["tiles_dir"], f"tiles_metadata_{slug}.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else []
    with rasterio.open(ortho_path) as src:
        entry = {
            "tile_id": window_key(window), "filename": f"{name}.jpg",
            "window_left": int(window.col_off), "window_top": int(window.row_off),
            "window_width": int(window.width), "window_height": int(window.height),
            "transform": list(src.window_transform(window))[:6], "crs": str(src.crs),
            "source_ortho": ortho_path,
        }
    meta = [m for m in meta if m["filename"] != entry["filename"]] + [entry]
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return name


def remove_training_tile(ortho_path, window, cfg):
    p = cfg["paths"]
    name = training_tile_name(ortho_path, window)
    for path in (os.path.join(p["tiles_dir"], f"{name}.jpg"), os.path.join(p["labels_dir"], f"{name}.txt")):
        if os.path.exists(path):
            os.remove(path)
    meta_path = os.path.join(p["tiles_dir"], f"tiles_metadata_{ortho_slug(ortho_path)}.json")
    if os.path.exists(meta_path):
        meta = [m for m in json.load(open(meta_path)) if m["filename"] != f"{name}.jpg"]
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def training_pool_summary(cfg):
    """Tiles etiquetados por ortomosaico (solo reales, sin sintéticos)."""
    from .common import load_tiles_metadata
    from .dataset import labeled_pairs
    p = cfg["paths"]
    meta = load_tiles_metadata(p["tiles_dir"])
    counts = {}
    for img, _ in labeled_pairs(p["tiles_dir"], p["labels_dir"]):
        m = meta.get(os.path.basename(img))
        key = m["ortho"] if m else "sin metadata"
        counts[key] = counts.get(key, 0) + 1
    return counts
