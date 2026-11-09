"""Preparación del ortomosaico: reproyección a UTM, lectura del perímetro del lote y recorte (sin UI)."""
import os
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window, from_bounds
from shapely.geometry import box

PERIMETER_FILTER = "Perímetro (*.zip *.shp *.gpkg *.geojson *.json *.kml)"
ROWS = 1024  # el recorte se escribe por franjas de filas para no cargar todo el orto en memoria
NAME_COLUMNS = ("lote", "nombre", "name", "lot", "parcela", "potrero")  # columnas que nombran al lote, por prioridad


def _lot_names(gdf):
    """Nombre de cada lote según la columna de atributos que lo identifique; si no hay, 'Lote N'."""
    columns = {str(c).lower(): c for c in gdf.columns}
    col = next((columns[c] for c in NAME_COLUMNS if c in columns), None)
    names = []
    for i, value in enumerate(gdf[col] if col else [None] * len(gdf)):
        text = "" if pd.isna(value) else str(value).strip().removesuffix(".0")
        names.append(f"Lote {text}" if text.isdigit() else text or f"Lote {i + 1}")
    return names


def _zip_layer(path):
    """Ruta para leer la única capa de un zip, esté en la raíz o en una carpeta. Con varias se avisa, en vez de
    leer la primera: recortaría con una capa que el usuario no eligió."""
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not n.endswith("/") and not n.startswith("__MACOSX/")]
    layers = [n for n in names if Path(n).suffix.lower() in (".shp", ".gpkg", ".geojson", ".kml")]
    # los .json pueden ser metadatos que acompañan a la capa: solo cuentan si no hay otra cosa
    layers = layers or [n for n in names if Path(n).suffix.lower() == ".json" and not n.endswith("-metadata.json")]
    zip_name = os.path.basename(path)
    if not layers:
        raise ValueError(f"{zip_name} no tiene ninguna capa (SHP, GPKG, GeoJSON o KML).")
    if len(layers) > 1:
        raise ValueError(f"{zip_name} tiene más de una capa; dejá solo la que quieras usar:\n" + "\n".join(layers))
    return f"/vsizip/{Path(path).as_posix()}/{layers[0]}"


def read_lots(path):
    """Lotes del archivo, uno por fila y en orden natural de nombre: columnas nombre, ha y geometry. Sin CRS se
    asume WGS84 si las coordenadas parecen lon/lat."""
    gdf = gpd.read_file(_zip_layer(path) if Path(path).suffix.lower() == ".zip" else path)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        raise ValueError(f"{os.path.basename(path)} no tiene polígonos.")
    if gdf.crs is None:
        x0, y0, x1, y1 = gdf.total_bounds
        if -180 <= x0 <= x1 <= 180 and -90 <= y0 <= y1 <= 90:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            raise ValueError(f"{os.path.basename(path)} no tiene sistema de coordenadas (falta el .prj).")
    geom = gdf.geometry.buffer(0)  # repara anillos mal cerrados
    ha = geom.to_crs(geom.estimate_utm_crs()).area / 1e4
    lots = gpd.GeoDataFrame({"nombre": _lot_names(gdf), "ha": ha.to_numpy()}, geometry=geom.to_numpy(), crs=gdf.crs)
    order = sorted(range(len(lots)), key=lambda i: [int(t) if t.isdigit() else t.lower()
                                                    for t in re.split(r"(\d+)", lots["nombre"].iloc[i])])
    return lots.iloc[order].reset_index(drop=True)


def lot_perimeter(path, lots, i):
    """Perímetro de trabajo: GeoDataFrame de una fila con el lote i. La columna `recorte` lo identifica para nombrar
    el orto recortado; un archivo de un solo lote conserva el nombre del archivo."""
    stem, lot = Path(path).stem, lots.iloc[i]
    if len(lots) == 1:
        nombre = recorte = stem
    else:
        nombre = lot["nombre"]
        recorte = f"{stem}__{re.sub(r'[^\w-]+', '_', nombre)[:40]}_{i}"  # con i: puede haber nombres repetidos
    return gpd.GeoDataFrame({"nombre": [nombre], "recorte": [recorte]}, geometry=[lot.geometry], crs=lots.crs)


def best_lot(lots, footprint, crs):
    """Lote al que corresponde el orto: (posición, parte del lote con datos), o None si ninguno lo toca.
    Puntaje = área cubierta x parte del lote cubierta: gana un lote entero frente a un vecino más grande cubierto a
    medias, y frente a un lote chico entero cuando el orto cubre también otro más grande."""
    geoms = lots.geometry.to_crs(crs).buffer(0)
    covered = geoms.intersection(footprint).area.to_numpy()
    if not covered.any():
        return None
    share = covered / geoms.area.to_numpy()
    i = int(np.argmax(covered * share))
    return i, float(share[i])


def perimeter_in(perim, crs):
    """Geometría del perímetro en el CRS del orto."""
    return perim.to_crs(crs).geometry.iloc[0]


def clipped_path(ortho_path, recorte, outputs_dir):
    """Ruta del orto recortado; `recorte` es el identificador que da lot_perimeter."""
    stem = Path(ortho_path).stem
    return os.path.join(outputs_dir, stem, "recortes", f"{stem}__{recorte}.tif")


def _open_rgba(path, width, height, dtype, crs, transform):
    """GeoTIFF RGBA de salida, por bloques y comprimido."""
    dst = rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=4, dtype=dtype, crs=crs,
                        transform=transform, tiled=True, blockxsize=512, blockysize=512, compress="deflate",
                        predictor=2, photometric="RGB", BIGTIFF="IF_SAFER")
    dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
    return dst


def _write_rgba(dst, rgb, keep, window):
    """Escribe una franja: lo que no está en `keep` queda negro y transparente."""
    rgb[:, ~keep] = 0
    alpha = np.where(keep, np.iinfo(rgb.dtype).max if rgb.dtype.kind in "ui" else 1, 0)
    dst.write(rgb, indexes=[1, 2, 3], window=window)
    dst.write(alpha.astype(rgb.dtype), 4, window=window)


def utm_path(ortho_path, outputs_dir):
    stem = Path(ortho_path).stem
    return os.path.join(outputs_dir, stem, f"{stem}_UTM.tif")


def reproject_to_utm(ortho_path, out_path, progress=None):
    """Escribe una copia del orto en su zona UTM (metros), por franjas y con la misma resolución aproximada. Las
    esquinas que quedan sin datos al rotar la imagen salen transparentes."""
    with rasterio.open(ortho_path) as src:
        utm = gpd.GeoSeries([box(*src.bounds)], crs=src.crs).estimate_utm_crs()
        crs = CRS.from_epsg(utm.to_epsg())
        transform, width, height = calculate_default_transform(src.crs, crs, src.width, src.height, *src.bounds)
        has_mask = ColorInterp.alpha in src.colorinterp or src.nodata is not None
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp.tif"
        with WarpedVRT(src, crs=crs, transform=transform, width=width, height=height,
                       resampling=Resampling.bilinear, add_alpha=not has_mask) as vrt, \
                _open_rgba(tmp, width, height, src.dtypes[0], crs, transform) as dst:
            for y in range(0, height, ROWS):
                window = Window(0, y, width, min(ROWS, height - y))
                _write_rgba(dst, vrt.read([1, 2, 3], window=window), vrt.dataset_mask(window=window) > 0, window)
                if progress:
                    progress(y + window.height, height)
    os.replace(tmp, out_path)
    return out_path


def clip_ortho(ortho_path, perim, out_path, progress=None):
    """Escribe el orto recortado al perímetro: ventana mínima que lo contiene y fuera del lote transparente
    (banda alfa). Si ya existe y es más nuevo que el orto y el perímetro, lo reutiliza."""
    with rasterio.open(ortho_path) as src:
        geom = perimeter_in(perim, src.crs)
        b = src.bounds
        if not geom.intersects(box(b.left, b.bottom, b.right, b.top)):
            raise ValueError("no cae dentro del ortomosaico.")
        win = from_bounds(*geom.bounds, transform=src.transform).round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp.tif"
        with _open_rgba(tmp, int(win.width), int(win.height), src.dtypes[0], src.crs, src.window_transform(win)) as dst:
            h = int(win.height)
            has_data = False
            for y in range(0, h, ROWS):
                rows = min(ROWS, h - y)
                sw = Window(win.col_off, win.row_off + y, win.width, rows)
                inside = geometry_mask([geom], out_shape=(rows, int(win.width)), transform=src.window_transform(sw),
                                       invert=True)
                keep = inside & (src.dataset_mask(window=sw) > 0)  # dentro del lote y con datos en el orto
                has_data = has_data or bool(keep.any())
                _write_rgba(dst, src.read([1, 2, 3], window=sw), keep, Window(0, y, win.width, rows))
                if progress:
                    progress(y + rows, h)
    if not has_data:  # toca el rectángulo del orto pero cae en su parte vacía
        os.remove(tmp)
        raise ValueError("no tiene datos en este ortomosaico.")
    os.replace(tmp, out_path)
    return out_path


def is_fresh(out_path, *sources):
    """El recorte existe y es posterior a todos los archivos de origen."""
    if not os.path.exists(out_path):
        return False
    t = os.path.getmtime(out_path)
    return all(t >= os.path.getmtime(s) for s in sources if os.path.exists(s))


def clip_to_perimeter(gdf, geom):
    """Recorta polígonos/cajas al perímetro y recalcula el área."""
    if gdf is None or not len(gdf):
        return gdf
    out = gpd.clip(gdf, geom)
    out = out[~out.geometry.is_empty & out.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if "area_m2" in out:
        out["area_m2"] = out.geometry.area
    return out.reset_index(drop=True)
