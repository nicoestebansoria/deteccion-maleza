"""Configuración y utilidades compartidas (tiles, labels YOLO, device)."""
import glob
import json
import os
from pathlib import Path

import numpy as np
import yaml
from rasterio.windows import Window


class Config(dict):
    """dict con acceso por atributo; rutas resueltas relativas al config.yaml."""

    def __getattr__(self, key):
        value = self[key]
        return Config(value) if isinstance(value, dict) else value


def load_config(path="config.yaml"):
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = Config(yaml.safe_load(f))
    root = path.parent
    cfg["paths"] = {k: str((root / v).resolve()) for k, v in cfg["paths"].items()}
    for o in cfg["orthos"]:
        o["path"] = str((root / o["path"]).resolve())
    w = cfg["predict"]["weights"]
    if w and not os.path.isabs(w):
        cfg["predict"]["weights"] = str((root / w).resolve())
    cfg["root"] = str(root)
    return cfg


def resolve_device(device):
    if device != "auto":
        return device
    import torch
    return 0 if torch.cuda.is_available() else "cpu"


# --- tiles ---

def iter_windows(width, height, tile_size, overlap):
    """Mismo recorrido que el notebook original (mantiene la numeración de tiles)."""
    step = tile_size - overlap
    for top in range(0, height, step):
        for left in range(0, width, step):
            w = min(tile_size, width - left)
            h = min(tile_size, height - top)
            if w <= 0 or h <= 0:
                continue
            yield Window(left, top, w, h)


def read_tile(src, window, tile_size):
    """Lee RGB de la ventana -> (tile HxWx3 paddeado a tile_size, fracción de píxeles negros)."""
    tile = np.moveaxis(src.read([1, 2, 3], window=window), 0, -1)
    black_frac = np.all(tile == 0, axis=-1).mean()
    if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
        padded = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
        padded[:tile.shape[0], :tile.shape[1], :] = tile
        tile = padded
    return tile, black_frac


def tile_name(prefix, tile_id):
    return f"tile_{prefix}_{tile_id:05d}" if prefix else f"tile_{tile_id:05d}"


def metadata_filename(prefix):
    return f"tiles_metadata_{prefix}.json" if prefix else "tiles_metadata.json"


def load_tiles_metadata(tiles_dir):
    """filename -> metadata, para todos los tiles_metadata*.json. Agrega la clave 'ortho'."""
    meta = {}
    for path in glob.glob(os.path.join(tiles_dir, "tiles_metadata*.json")):
        stem = Path(path).stem
        ortho = stem[len("tiles_metadata_"):] if stem != "tiles_metadata" else "o1"
        with open(path) as f:
            for m in json.load(f):
                m["ortho"] = ortho
                meta[m["filename"]] = m
    return meta


# --- labels YOLO ---

def load_yolo_labels(label_path):
    boxes = []
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, cx, cy, w, h = map(float, parts)
                boxes.append((cx, cy, w, h))
    return boxes


def write_yolo_labels(label_path, boxes):
    with open(label_path, "w") as f:
        f.write("\n".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes))


def yolo_to_px(box, tile_w, tile_h):
    cx, cy, w, h = box
    return (int((cx - w / 2) * tile_w), int((cy - h / 2) * tile_h),
            int((cx + w / 2) * tile_w), int((cy + h / 2) * tile_h))


def px_to_yolo(x1, y1, x2, y2, tile_w, tile_h):
    return ((x1 + x2) / 2 / tile_w, (y1 + y2) / 2 / tile_h,
            (x2 - x1) / tile_w, (y2 - y1) / tile_h)
