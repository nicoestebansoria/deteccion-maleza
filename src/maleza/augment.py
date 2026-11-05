"""Copy-paste augmentation (celdas 18 y 18b del notebook), corregida:

- define los donantes (`all_donors`) y `boxes_overlap`, que faltaban en el notebook;
- la caja del label sintético es el bbox de la planta segmentada, no el recorte con margen;
- solo usa tiles de train como fondo y como donantes (sin fuga hacia val).
"""
import os
import random

import cv2
import numpy as np

from .common import load_yolo_labels, px_to_yolo, write_yolo_labels, yolo_to_px


def boxes_overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def extract_plant(img_bgr, x1, y1, x2, y2, margin_min=15, margin_ratio=0.35, iters=5):
    """GrabCut alrededor del box -> (recorte BGR, máscara 0/255) ajustados al blob principal."""
    h_img, w_img = img_bgr.shape[:2]
    box_w, box_h = x2 - x1, y2 - y1
    margin = max(margin_min, int((box_w ** 2 + box_h ** 2) ** 0.5 * margin_ratio))

    mx1, my1 = max(0, x1 - margin), max(0, y1 - margin)
    mx2, my2 = min(w_img, x2 + margin), min(h_img, y2 + margin)
    crop = img_bgr[my1:my2, mx1:mx2]
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return None, None

    rect = (max(0, x1 - mx1), max(0, y1 - my1), max(1, box_w), max(1, box_h))
    mask = np.zeros(crop.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None, None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # quedarse con el blob más grande y recortar a su bbox: ese es el box del label
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    fg = np.zeros_like(fg)
    cv2.drawContours(fg, [largest], -1, 255, thickness=cv2.FILLED)
    bx, by, bw, bh = cv2.boundingRect(largest)
    if bw < 5 or bh < 5:
        return None, None
    return crop[by:by + bh, bx:bx + bw].copy(), fg[by:by + bh, bx:bx + bw]


def find_donors(pairs, area_percentile=85, min_cluster_boxes=5):
    """Clasifica tiles etiquetados -> (donantes [(img_path, box)], tiles vacíos de fondo)."""
    tile_boxes = {img: load_yolo_labels(lbl) for img, lbl in pairs}
    empty_tiles = [img for img, boxes in tile_boxes.items() if not boxes]
    all_areas = [b[2] * b[3] for boxes in tile_boxes.values() for b in boxes]
    area_threshold = np.percentile(all_areas, area_percentile) if all_areas else 0

    large_clumps, dense_clusters = [], []
    for img, boxes in tile_boxes.items():
        if len(boxes) >= min_cluster_boxes:
            dense_clusters.extend((img, b) for b in boxes)
        large_clumps.extend((img, b) for b in boxes if b[2] * b[3] >= area_threshold)

    print(f"Manchones grandes: {len(large_clumps)} | Boxes en clusters densos: "
          f"{len(dense_clusters)} | Tiles vacíos de fondo: {len(empty_tiles)}")
    # set: un box puede ser manchón grande y estar en un cluster denso a la vez
    return sorted(set(large_clumps + dense_clusters)), empty_tiles


def generate_synthetic(pairs, images_out, labels_out, cfg):
    """Genera tiles sintéticos pegando donantes sobre tiles vacíos. Devuelve cuántos generó."""
    rng = random.Random(cfg["seed"])
    donors, empty_tiles = find_donors(pairs, cfg["area_percentile"], cfg["min_cluster_boxes"])
    if not donors or not empty_tiles:
        print("Sin donantes o sin tiles vacíos en train: no se generan sintéticos.")
        return 0

    img_cache = {}

    def imread(path):
        if path not in img_cache:
            img_cache[path] = cv2.imread(path)
        return img_cache[path]

    generated = 0
    for synth_idx in range(cfg["n_synthetic"]):
        background = imread(rng.choice(empty_tiles))
        if background is None:
            continue
        canvas = background.copy()
        tile_h, tile_w = canvas.shape[:2]
        placed = []
        n_target = rng.randint(cfg["min_donors_per_tile"], cfg["max_donors_per_tile"])
        draws = 0

        while len(placed) < n_target and draws < n_target * cfg["max_donor_draws_multiplier"]:
            draws += 1
            donor_path, donor_box = rng.choice(donors)
            donor_img = imread(donor_path)
            if donor_img is None:
                continue
            dh, dw = donor_img.shape[:2]
            x1, y1, x2, y2 = yolo_to_px(donor_box, dw, dh)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(dw, x2), min(dh, y2)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            crop, mask = extract_plant(donor_img, x1, y1, x2, y2)
            if crop is None:
                continue
            # vista aérea: cualquier flip / rotación de 90° es válida
            if rng.random() < 0.5:
                crop, mask = cv2.flip(crop, 1), cv2.flip(mask, 1)
            if rng.random() < 0.5:
                crop, mask = cv2.flip(crop, 0), cv2.flip(mask, 0)
            k = rng.randint(0, 3)
            crop, mask = np.rot90(crop, k).copy(), np.rot90(mask, k).copy()

            ch, cw = crop.shape[:2]
            if ch >= tile_h or cw >= tile_w:
                continue  # donante demasiado grande para este lienzo

            for _ in range(cfg["max_placement_attempts"]):
                px, py = rng.randint(0, tile_w - cw), rng.randint(0, tile_h - ch)
                new_box = (px, py, px + cw, py + ch)
                if any(boxes_overlap(new_box, pb) for pb in placed):
                    continue
                roi = canvas[py:py + ch, px:px + cw]
                m = mask.astype(bool)
                roi[m] = crop[m]
                placed.append(new_box)
                break

        if not placed:
            continue
        name = f"tile_aug_{synth_idx:05d}"
        cv2.imwrite(os.path.join(images_out, f"{name}.jpg"), canvas)
        write_yolo_labels(os.path.join(labels_out, f"{name}.txt"),
                          [px_to_yolo(*b, tile_w, tile_h) for b in placed])
        generated += 1

    print(f"Tiles sintéticos generados: {generated} de {cfg['n_synthetic']} intentados")
    return generated
