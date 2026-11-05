"""Inferencia sobre el ortomosaico completo + dedup + georreferenciación (celdas 10 y 11).

A diferencia del notebook, lee el .tif directamente con ventana deslizante (no hacen falta
los JPG ni el tiles_metadata.json) y deduplica por solape de cajas en vez de DBSCAN sobre
centros, que encadenaba plantas vecinas de una misma fila en una sola detección.
"""
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box as shapely_box

from .common import iter_windows, read_tile, resolve_device


def _batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def merge_tiles_nms(boxes, scores, truncated, iou_thr, ios_thr):
    """NMS greedy entre tiles solapados.

    Una caja se descarta si tiene IoU > iou_thr con otra de mayor confianza, o si está
    cortada por el borde de su tile y la intersección cubre > ios_thr de la caja más chica
    (la misma planta completa en el tile vecino). El criterio IoS se limita a cajas cortadas
    para que un manchón grande no absorba plantas chicas que estén adentro.
    """
    # primero las cajas completas (por confianza), después las cortadas: ante un duplicado gana la completa
    order = np.lexsort((-scores, truncated))
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        ix1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        iy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        ix2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        iy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        ios = inter / np.maximum(np.minimum(areas[i], areas[rest]), 1e-9)
        cut = truncated[i] | truncated[rest]
        drop = (iou > iou_thr) | (cut & (ios > ios_thr))
        order = rest[~drop]
    return np.array(keep, dtype=int)


EDGE_PX = 2  # caja a menos de esto del borde interno del tile = probablemente cortada


def truncated_flags(xyxy, window, raster_w, raster_h, edge_px=EDGE_PX):
    """Cajas (coords del tile) que tocan un borde del tile que no es borde del raster."""
    return (
        ((xyxy[:, 0] <= edge_px) & (window.col_off > 0))
        | ((xyxy[:, 1] <= edge_px) & (window.row_off > 0))
        | ((xyxy[:, 2] >= window.width - edge_px) & (window.col_off + window.width < raster_w))
        | ((xyxy[:, 3] >= window.height - edge_px) & (window.row_off + window.height < raster_h))
    )


def boxes_to_gdf(boxes_px, scores, truncated, transform, crs, source="modelo"):
    """Cajas en píxeles del ortomosaico -> GeoDataFrame (guarda también las coords en px)."""
    geoms = []
    for x1, y1, x2, y2 in boxes_px:
        gx1, gy1 = transform * (x1, y1)
        gx2, gy2 = transform * (x2, y2)
        geoms.append(shapely_box(min(gx1, gx2), min(gy1, gy2), max(gx1, gx2), max(gy1, gy2)))
    boxes_px = np.asarray(boxes_px, dtype=float).reshape(-1, 4)
    return gpd.GeoDataFrame({
        "confidence": np.asarray(scores, dtype=float),
        "source": source,
        "px_x1": boxes_px[:, 0], "px_y1": boxes_px[:, 1], "px_x2": boxes_px[:, 2], "px_y2": boxes_px[:, 3],
        "truncated": np.asarray(truncated, dtype=bool),
    }, geometry=geoms, crs=crs)


def detect_ortho(model, ortho_path, cfg, progress=None, conf=None):
    """Corre el modelo sobre todo el ortomosaico -> (GeoDataFrame de cajas, n detecciones crudas).

    progress: callback opcional progress(hechos, total) para mostrar avance (ej. en la app).
    conf: umbral de confianza (default cfg.predict.conf). La app usa uno bajo y filtra después.
    """
    t, pr = cfg["tiling"], cfg["predict"]
    tile_size = t["tile_size"]
    conf = pr["conf"] if conf is None else conf
    device = resolve_device(pr["device"])
    all_boxes, all_scores, all_truncated = [], [], []

    with rasterio.open(ortho_path) as src:
        windows = list(iter_windows(src.width, src.height, tile_size, t["overlap"]))
        transform, crs = src.transform, src.crs

        done = 0
        for batch_windows in _batched(windows, pr["batch"]):
            batch = []
            for window in batch_windows:
                tile, black_frac = read_tile(src, window, tile_size)
                if black_frac <= pr["nodata_threshold"]:
                    batch.append((window, tile))
            done += len(batch_windows)
            if progress:
                progress(done, len(windows))
            else:
                print(f"\r  {done} tiles procesados de {len(windows)}", end="", flush=True)
            if not batch:
                continue
            # ultralytics interpreta arrays numpy como BGR (igual que cv2.imread en train)
            imgs = [np.ascontiguousarray(tile[:, :, ::-1]) for _, tile in batch]
            results = model.predict(imgs, conf=conf, imgsz=tile_size, device=device, verbose=False)
            for (window, _), r in zip(batch, results):
                if len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                # descartar cajas que caen en el padding del tile de borde
                valid = (xyxy[:, 0] < window.width) & (xyxy[:, 1] < window.height)
                xyxy = xyxy[valid]
                xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, window.width)
                xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, window.height)
                truncated = truncated_flags(xyxy, window, src.width, src.height)
                xyxy[:, [0, 2]] += window.col_off
                xyxy[:, [1, 3]] += window.row_off
                all_boxes.append(xyxy)
                all_scores.append(r.boxes.conf.cpu().numpy()[valid])
                all_truncated.append(truncated)
        if not progress:
            print()

    if not all_boxes:
        return boxes_to_gdf(np.zeros((0, 4)), [], [], transform, crs), 0

    boxes_px, scores = np.concatenate(all_boxes), np.concatenate(all_scores)
    truncated = np.concatenate(all_truncated)
    keep = merge_tiles_nms(boxes_px, scores, truncated, pr["dedup_iou"], pr["dedup_ios"])
    return boxes_to_gdf(boxes_px[keep], scores[keep], truncated[keep], transform, crs), len(boxes_px)


def load_model(weights):
    from ultralytics import YOLO
    return YOLO(weights, task="detect")


def inspect_tile(model, tile_path, conf=0.05):
    """Diagnóstico (celda 14): confianzas crudas en un tile puntual con umbral muy bajo."""
    for r in model.predict(tile_path, conf=conf, verbose=False):
        for b in r.boxes:
            print(f"Confianza: {float(b.conf[0]):.3f} — box: {[round(v) for v in b.xyxy[0].tolist()]}")
