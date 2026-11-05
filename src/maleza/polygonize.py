"""Poligonización de detecciones con GrabCut + export (celdas 12 y 13 del notebook)."""
import math
import os

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.windows import Window, from_bounds
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union


def _grabcut_mask(img_bgr, rect, iters, max_px):
    """Máscara 0/1 de GrabCut. GrabCut escala mal con el tamaño: recortes grandes se procesan reducidos."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_px / max(h, w))
    if scale < 1.0:
        small = cv2.resize(img_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        small_rect = tuple(max(1, int(v * scale)) if i >= 2 else int(v * scale) for i, v in enumerate(rect))
    else:
        small, small_rect = img_bgr, rect
    mask = np.zeros(small.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, small_rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    return cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST) if scale < 1.0 else fg


def _exg_mask(img_bgr, rect, threshold):
    """Máscara 0/1 de píxeles verdes (ExG = 2g - r - b sobre cromaticidades) dentro del rect."""
    b, g, r = (img_bgr[..., i].astype(np.float32) for i in range(3))
    s = b + g + r + 1e-6
    exg = (2 * g - r - b) / s
    x, y, w, h = rect
    fg = np.zeros(img_bgr.shape[:2], np.uint8)
    fg[y:y + h, x:x + w] = (exg[y:y + h, x:x + w] > threshold).astype(np.uint8)
    return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _largest_polygon(fg, wt, min_area_px):
    if fg is None:
        return None
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area_px:
        return None
    largest = cv2.approxPolyDP(largest, 0.5, True)
    if len(largest) < 3:
        return None
    poly = Polygon([wt * (float(pt[0][0]), float(pt[0][1])) for pt in largest])
    if not poly.is_valid:
        poly = poly.buffer(0)
    return None if poly.is_empty or poly.area == 0 else poly


def polygonize(gdf_boxes, ortho_path, cfg, progress=None):
    """Contorno de cada detección: GrabCut -> si falla, verde (ExG) dentro de la caja -> si falla, la caja.

    Ninguna detección se pierde; la columna 'metodo' indica cómo se obtuvo cada polígono.
    progress: callback opcional progress(hechos, total).
    """
    pc = cfg["polygonize"]
    polygons, metodos = [], {"grabcut": 0, "exg": 0, "caja": 0}

    with rasterio.open(ortho_path) as src:
        full = Window(0, 0, src.width, src.height)

        for n, row in enumerate(gdf_boxes.itertuples(), 1):
            if progress and (n % 20 == 0 or n == len(gdf_boxes)):
                progress(n, len(gdf_boxes))
            minx, miny, maxx, maxy = row.geometry.bounds
            margin_m = max(pc["margin_min_m"], math.hypot(maxx - minx, maxy - miny) * pc["margin_ratio"])
            window = from_bounds(minx - margin_m, miny - margin_m, maxx + margin_m, maxy + margin_m,
                                 transform=src.transform)
            # redondear a píxeles enteros y recortar al raster (detecciones en el borde)
            poly, metodo = None, "caja"
            try:
                window = window.round_offsets().round_lengths().intersection(full)
                crop = src.read([1, 2, 3], window=window)
            except Exception:
                crop = None

            if crop is not None and crop.shape[1] >= 10 and crop.shape[2] >= 10:
                wt = src.window_transform(window)
                img_bgr = np.ascontiguousarray(np.transpose(crop, (1, 2, 0))[:, :, ::-1]).astype(np.uint8)
                h, w = img_bgr.shape[:2]
                # rect del box original dentro del recorte (en píxeles)
                inv = ~wt
                bx1, by1 = inv * (minx, maxy)
                bx2, by2 = inv * (maxx, miny)
                rx1, ry1 = int(max(0, min(bx1, bx2))), int(max(0, min(by1, by2)))
                rx2, ry2 = int(min(w, max(bx1, bx2))), int(min(h, max(by1, by2)))
                rect = (rx1, ry1, max(1, rx2 - rx1), max(1, ry2 - ry1))

                poly = _largest_polygon(_grabcut_mask(img_bgr, rect, pc["grabcut_iter"], pc.get("max_grabcut_px", 256)),
                                        wt, pc["min_blob_area_px"])
                metodo = "grabcut"
                if poly is None:
                    poly = _largest_polygon(_exg_mask(img_bgr, rect, pc.get("exg_threshold", 0.05)),
                                            wt, pc["min_blob_area_px"])
                    metodo = "exg"
            if poly is None:
                poly, metodo = row.geometry, "caja"

            metodos[metodo] += 1
            polygons.append({"geometry": poly, "confidence": row.confidence, "area_m2": poly.area,
                             "source": getattr(row, "source", "modelo"), "metodo": metodo})

    print(f"Polígonos: {len(polygons)} | GrabCut: {metodos['grabcut']} · verde (ExG): {metodos['exg']} · "
          f"caja: {metodos['caja']}")
    return gpd.GeoDataFrame(polygons, geometry="geometry", crs=gdf_boxes.crs) if polygons         else gpd.GeoDataFrame({"confidence": [], "area_m2": [], "source": [], "metodo": []},
                              geometry=[], crs=gdf_boxes.crs)


def green_components(ortho_path, cfg):
    """Máscara de verde (ExG) de todo el ortomosaico, separada en manchas conexas. Con green_resolution_m = 0
    se usa la resolución original; si no, el orto se promedia a esa resolución.

    Devuelve (labels int32, transform, área en m², ExG medio y ExG pico de cada mancha, indexados por label).
    En barbecho todo lo verde es maleza: las manchas sirven para completar manchones que el modelo
    detecta solo en partes.
    """
    pc = cfg["polygonize"]
    with rasterio.open(ortho_path) as src:
        res = abs(src.transform.a)
        factor = max(1, round((pc.get("green_resolution_m") or 0) / res))
        h, w = src.height // factor, src.width // factor
        rgb = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.average)
        valid = src.dataset_mask(out_shape=(h, w)) > 0
        transform = src.transform * src.transform.scale(src.width / w, src.height / h)

    # escala a 8 bits para que el verde absoluto (2G-R-B) se mida igual en ortos de 16 bits
    scale = 1.0 if rgb.dtype == np.uint8 else 255.0 / max(float(np.percentile(rgb[:, ::16, ::16], 99.5)), 1.0)
    exg = np.zeros((h, w), np.float32)
    green_abs = np.zeros((h, w), np.float32)
    for y in range(0, h, 1024):  # por bloques de filas: el ExG en float32 de todo el orto pesa mucho
        r, g, b = (rgb[i, y:y + 1024].astype(np.float32) * scale for i in range(3))
        exg[y:y + 1024] = (2 * g - r - b) / (r + g + b + 1e-6)
        green_abs[y:y + 1024] = 2 * g - r - b
    del rgb
    fg = ((exg > pc.get("exg_threshold", 0.05)) & valid).astype(np.uint8)
    # motas: grupos de píxeles verdes más chicos que green_speck_m2 se borran antes del cierre. A resolución
    # original el rastrojo deja muchos píxeles sueltos que el cierre uniría en manchones falsos; a diferencia
    # de un filtro de mediana, no recorta el borde de las plantas chicas (que así se perdían).
    px = abs(transform.a)
    speck = (pc.get("green_speck_m2", 0) or 0) / (px * px)
    if speck > 1:
        _, lab, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        keep = st[:, cv2.CC_STAT_AREA] >= speck
        keep[0] = False
        fg = keep[lab].astype(np.uint8)
        del lab

    k = max(1, round(pc.get("green_close_m", 0.3) / px)) | 1  # cierre: une plantas de un mismo manchón
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    npx = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    areas = npx * px * px
    areas[0] = 0
    # ExG medio de cada mancha: separa plantas (verde intenso) de rastrojo que apenas pasa el umbral
    mean_exg = np.bincount(labels.ravel(), weights=exg.ravel(), minlength=n) / np.maximum(npx, 1)
    # pico de verde ABSOLUTO (2G-R-B, escala 0-255): una planta tiene un núcleo de verde intenso. No se usa el ExG
    # normalizado porque en las sombras oscuras del rastrojo el divisor (R+G+B) es chico y un leve tinte verdoso
    # da valores altos: en la revisión manual separó sí/no con AUC 0.90, el absoluto con 0.98.
    # Promediado a green_peak_window_m (10 cm) para que un solo píxel ruidoso no defina el pico.
    del exg
    kp = max(1, round(pc.get("green_peak_window_m", 0.1) / px))
    if kp > 1:
        green_abs = cv2.blur(green_abs, (kp, kp))
    peak_exg = np.full(n, -1e9, np.float32)
    np.maximum.at(peak_exg, labels.ravel(), green_abs.ravel())
    del green_abs
    print(f"Verde (ExG): {n - 1} manchas a {px:.2f} m/px | {areas.sum():.0f} m²")
    return labels, transform, areas, mean_exg, peak_exg


def polygonize_green(gdf_boxes, ortho_path, cfg, components=None, include_orphans=False, keep_no_green=False,
                     progress=None):
    """Contorno por manchón verde: cada detección toma las manchas verdes que toca (enteras, aunque
    excedan la caja) y las detecciones que comparten mancha se unen en un solo polígono.

    Manchas con verde pico (2G-R-B) < green_min_peak se descartan (rastrojo o sombras). Las manchas sin
    detección exigen green_orphan_min_peak (más alto): sin el modelo que las confirme, el verde tenue suele
    ser rastrojo (revisión manual 2026-11-03: con pico >= 30 queda el 96% de la maleza y el 5% de lo que no es).
    Detecciones sin verde: se descartan (en barbecho, sin verde no hay maleza) o, con keep_no_green,
    van a polygonize() clásico (GrabCut / caja).
    include_orphans: agrega manchas >= green_orphan_min_m2 que ninguna detección tocó (source 'exg').
    components: resultado de green_components() (la app lo cachea).
    """
    pc = cfg["polygonize"]
    labels, transform, areas, mean_exg, peak_exg = components or green_components(ortho_path, cfg)
    h, w = labels.shape
    # manchas válidas: tamaño mínimo y verde medio suficiente
    valid = (areas >= pc.get("green_min_area_m2", 0.1)) & (peak_exg >= pc.get("green_min_peak", 0))
    valid[0] = False
    inv = ~transform

    # union-find de detecciones que comparten mancha
    parent = list(range(len(gdf_boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}            # label -> detección
    det_labels = []       # labels tocados por cada detección
    for i, geom in enumerate(gdf_boxes.geometry):
        minx, miny, maxx, maxy = geom.bounds
        c1, r1 = inv * (minx, maxy)
        c2, r2 = inv * (maxx, miny)
        c1, r1 = max(0, int(math.floor(c1))), max(0, int(math.floor(r1)))
        c2, r2 = min(w, int(math.ceil(c2))), min(h, int(math.ceil(r2)))
        found = np.unique(labels[r1:r2, c1:c2]) if c2 > c1 and r2 > r1 else np.array([], np.int32)
        found = [int(l) for l in found if valid[l]]
        det_labels.append(found)
        for l in found:
            if l in owner:
                parent[find(i)] = find(owner[l])
            else:
                owner[l] = i
    if progress:
        progress(1, 3)

    # raster de grupo: cada label -> id de grupo (raíz + 1); 0 = fondo / sin dueño
    lut = np.zeros(len(areas), np.int32)
    for l, i in owner.items():
        lut[l] = find(i) + 1
    orphan_min = pc.get("green_orphan_min_m2", 1.0)
    orphan_base = len(gdf_boxes) + 1
    if include_orphans:
        orphan_peak = max(pc.get("green_orphan_min_peak", 0), pc.get("green_min_peak", 0))
        orphans = np.flatnonzero((lut == 0) & valid & (areas >= orphan_min) & (peak_exg >= orphan_peak))
        lut[orphans] = orphan_base + np.arange(len(orphans), dtype=np.int32)
    groups = lut[labels]
    # ExG medio por grupo, ponderado por área de cada mancha
    w_area = np.bincount(lut, weights=areas, minlength=lut.max() + 1)
    group_exg = np.bincount(lut, weights=areas * mean_exg, minlength=lut.max() + 1) / np.maximum(w_area, 1e-9)
    group_peak = np.full(lut.max() + 1, -1e9, np.float32)
    np.maximum.at(group_peak, lut, peak_exg)

    geoms = {}
    for geom, v in shapes(groups, mask=groups > 0, transform=transform, connectivity=8):
        geoms.setdefault(int(v), []).append(shape(geom))
    if progress:
        progress(2, 3)

    tol = abs(transform.a) / 2
    max_expand = pc.get("green_max_expand_m", 5.0)
    boxes_geom = list(gdf_boxes.geometry)
    rows = []
    members = {}
    for i in range(len(gdf_boxes)):
        if det_labels[i]:
            members.setdefault(find(i) + 1, []).append(i)
    conf = gdf_boxes["confidence"].to_numpy()
    src_col = gdf_boxes["source"].to_numpy() if "source" in gdf_boxes else np.array(["modelo"] * len(gdf_boxes))
    for gid, parts in geoms.items():
        poly = unary_union(parts).simplify(tol)
        if gid >= orphan_base:
            rows.append({"geometry": poly, "confidence": 0.0, "area_m2": poly.area, "source": "exg",
                         "metodo": "verde_sin_deteccion", "n_det": 0, "exg_medio": round(float(group_exg[gid]), 3),
                         "verde_pico": int(round(float(group_peak[gid])))})
        else:
            idx = members[gid]
            if max_expand > 0:  # evita que una caja se lleve una franja verde larga (bordes, alambrados)
                poly = poly.intersection(unary_union([boxes_geom[i].buffer(max_expand) for i in idx]))
                if poly.is_empty:
                    continue
            rows.append({"geometry": poly, "confidence": float(conf[idx].max()), "area_m2": poly.area,
                         "source": "manual" if (src_col[idx] == "manual").any() else "modelo",
                         "metodo": "verde", "n_det": len(idx), "exg_medio": round(float(group_exg[gid]), 3),
                         "verde_pico": int(round(float(group_peak[gid])))})

    sin_verde = [i for i in range(len(gdf_boxes)) if not det_labels[i]] if keep_no_green else []
    n_descartadas = sum(1 for d in det_labels if not d) if not keep_no_green else 0
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf_boxes.crs) if rows else None
    if sin_verde:
        rest = polygonize(gdf_boxes.iloc[sin_verde], ortho_path, cfg)
        rest["n_det"] = 1
        gdf = rest if gdf is None else gpd.GeoDataFrame(
            pd.concat([gdf, rest], ignore_index=True), geometry="geometry", crs=gdf_boxes.crs)
    if progress:
        progress(3, 3)
    if gdf is None:
        return gpd.GeoDataFrame({"confidence": [], "area_m2": [], "source": [], "metodo": [], "n_det": []},
                                geometry=[], crs=gdf_boxes.crs)
    n_orph = int((gdf["metodo"] == "verde_sin_deteccion").sum())
    print(f"Manchones verdes: {len(gdf) - n_orph - len(sin_verde)} | detecciones sin verde: "
          f"{len(sin_verde) or n_descartadas} ({'GrabCut/caja' if keep_no_green else 'descartadas'}) | "
          f"manchas sin detección: {n_orph}")
    return gdf


def ortho_footprint(ortho_path, cell_m=0.4):
    """Polígono del área con datos del orto (sin el fondo transparente/negro), a cell_m de resolución."""
    with rasterio.open(ortho_path) as src:
        f = max(1, round(cell_m / abs(src.transform.a)))
        h, w = max(1, src.height // f), max(1, src.width // f)
        m = (src.dataset_mask(out_shape=(h, w)) > 0).astype(np.uint8)
        transform = src.transform * src.transform.scale(src.width / w, src.height / h)
    return unary_union([shape(g) for g, v in shapes(m, mask=m > 0, transform=transform)])


def clip_border(gdf, footprint, margin_m):
    """Recorta los polígonos a margin_m hacia adentro del borde del orto: ahí suele haber verde que no es maleza
    del lote (alambrado, banquina, cultivo vecino). Recorta en vez de descartar, para no perder manchones reales
    que llegan cerca del borde."""
    if margin_m <= 0 or not len(gdf):
        return gdf
    out = gpd.clip(gdf, footprint.buffer(-margin_m))
    out = out[~out.geometry.is_empty].copy()
    out["area_m2"] = out.geometry.area
    return out.reset_index(drop=True)


def export(gdf_boxes, gdf_polygons, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gdf_boxes.to_file(os.path.join(out_dir, "malezas_detecciones.gpkg"), driver="GPKG")
    gdf_polygons.to_file(os.path.join(out_dir, "malezas_poligonos.gpkg"), driver="GPKG")
    gdf_polygons.to_file(os.path.join(out_dir, "malezas_poligonos.geojson"), driver="GeoJSON")
    if len(gdf_polygons):
        gdf_polygons.to_file(os.path.join(out_dir, "malezas_poligonos.shp"), driver="ESRI Shapefile")

    print(f"Resultados en: {out_dir}")
    print(f"  Total de polígonos: {len(gdf_polygons)}")
    if len(gdf_polygons):
        print(f"  Área total de maleza mapeada: {gdf_polygons['area_m2'].sum():.2f} m²")
        print(f"  Área promedio por planta: {gdf_polygons['area_m2'].mean():.3f} m²")
        print(f"  Confianza promedio: {gdf_polygons['confidence'].mean():.2f}")
