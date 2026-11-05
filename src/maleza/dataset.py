"""Split train/val espacial + sintéticos solo en train + data.yaml (celdas 7, 8 y 19)."""
import glob
import os
import random
import shutil
from collections import defaultdict

from .augment import generate_synthetic
from .common import load_tiles_metadata


def labeled_pairs(tiles_dir, labels_dir):
    """Pares (tile, label) reales. Ignora tiles sintéticos viejos (tile_aug_*) de Drive."""
    pairs = []
    for tile_path in sorted(glob.glob(os.path.join(tiles_dir, "*.jpg"))):
        name = os.path.splitext(os.path.basename(tile_path))[0]
        label_path = os.path.join(labels_dir, f"{name}.txt")
        if not name.startswith("tile_aug_") and os.path.exists(label_path):
            pairs.append((tile_path, label_path))
    return pairs


def spatial_split(pairs, tiles_meta, val_ratio, seed, block_px):
    """Agrupa tiles en bloques espaciales y asigna bloques enteros a val hasta cubrir val_ratio."""
    groups = defaultdict(list)
    for pair in pairs:
        m = tiles_meta.get(os.path.basename(pair[0]))
        if m:
            # bloque por coordenadas del centro del tile (no por ortomosaico): tiles de distintos
            # ortos/vuelos sobre la misma zona del lote caen en el mismo bloque y no se filtran a val
            a, _, c, _, e, f = m["transform"][:6]
            block_m = block_px * abs(a)
            cx, cy = c + a * m["window_width"] / 2, f + e * m["window_height"] / 2
            key = (m["crs"], int(cx // block_m), int(cy // block_m))
        else:
            key = ("sin_meta", pair[0])
        groups[key].append(pair)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_val_target = round(len(pairs) * val_ratio)
    train, val = [], []
    for key in keys:
        (val if len(val) < n_val_target else train).extend(groups[key])
    print(f"Split espacial: {len(groups)} bloques | Train: {len(train)} | Val: {len(val)}")
    return train, val


def build_dataset(cfg, dataset_dir=None):
    """dataset_dir: default paths.dataset_dir. La app usa uno por corrida para no pisar
    el dataset de un entrenamiento en curso."""
    p = cfg["paths"]
    dataset_dir = dataset_dir or p["dataset_dir"]
    pairs = labeled_pairs(p["tiles_dir"], p["labels_dir"])
    print(f"Pares tile+label reales: {len(pairs)}")
    if not pairs:
        raise SystemExit(f"No hay tiles etiquetados en {p['tiles_dir']} / {p['labels_dir']}")

    t, s = cfg["tiling"], cfg["split"]
    block_px = (t["tile_size"] - t["overlap"]) * s["block_tiles"]
    train, val = spatial_split(pairs, load_tiles_metadata(p["tiles_dir"]),
                               s["val_ratio"], s["seed"], block_px)

    # limpiar dataset/ previo para no dejar residuos de splits anteriores
    for split, split_pairs in [("train", train), ("val", val)]:
        for kind in ("images", "labels"):
            d = os.path.join(dataset_dir, kind, split)
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
        for tile_path, label_path in split_pairs:
            shutil.copy(tile_path, os.path.join(dataset_dir, "images", split))
            shutil.copy(label_path, os.path.join(dataset_dir, "labels", split))

    if cfg["augment"]["enabled"]:
        generate_synthetic(train, os.path.join(dataset_dir, "images", "train"),
                           os.path.join(dataset_dir, "labels", "train"), cfg["augment"])

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {dataset_dir}\ntrain: images/train\nval: images/val\n\n"
                f"nc: 1\nnames: ['maleza']\n")
    print(f"data.yaml -> {yaml_path}")
    return yaml_path
