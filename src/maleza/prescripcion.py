"""Mapa de prescripción a partir de los polígonos de maleza y export a formatos de pulverizadoras y drones (sin UI).

Zonas: "maleza" (polígonos con buffer +pos, disueltos y buffer -neg) y, opcional, "fondo" (el resto del lote).
Formatos de export:
- Shapefile (WGS84, campo rx): John Deere (Operations Center / Gen4 por USB), Trimble, Raven, Ag Leader y la
  mayoría de los monitores. También se entrega en .zip, que es lo que piden las plataformas web.
- ISOXML (ISO 11783-10, TASKDATA/TASKDATA.XML + grilla tipo 1): terminales ISOBUS (Case IH, New Holland, AGCO,
  CLAAS, Amazone, Hardi, Topcon, Müller, CCI...).
- DJI Agras: carpeta DJI/ con el límite en shapefile y el mapa de dosis en GeoTIFF (como lo exporta Pix4Dfields).
"""
import os
import zipfile
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.ops import unary_union

# unidad -> (DDI ISOBUS de dosis por área, factor de la unidad a la del DDI)
# DDI 1: Setpoint Volume Per Area Application Rate (mm³/m²; 1 L/ha = 100). DDI 6: Setpoint Mass Per Area (mg/m²;
# 1 kg/ha = 100).
UNITS = {"L/ha": (1, 100.0), "kg/ha": (6, 100.0)}
COLS = ["zona", "rx", "unidad", "ha", "geometry"]


def _polygonal(g):
    """Solo la parte de área: intersection/difference pueden devolver colecciones con líneas o puntos."""
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    return unary_union([p for p in getattr(g, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")])


def build_prescription(polys, boundary, buf_pos, buf_neg, rx_weed, rx_bg, background, unit="L/ha"):
    """GeoDataFrame con una fila por zona (multipolígono) en el CRS de los polígonos.

    boundary: polígono del lote (perímetro cargado o área con datos del orto); recorta todo y define el fondo."""
    crs = polys.crs
    geoms = list(polys.geometry)
    weed = unary_union([g.buffer(buf_pos) for g in geoms] if buf_pos > 0 else geoms) if geoms else None
    if weed is not None and buf_neg > 0:
        weed = weed.buffer(-buf_neg)
    if weed is not None:
        weed = _polygonal(weed.intersection(boundary))
    rows = []
    if weed is not None and not weed.is_empty:
        rows.append({"zona": "maleza", "rx": float(rx_weed), "geometry": weed})
    if background:
        bg = _polygonal(boundary.difference(weed)) if rows else boundary
        if not bg.is_empty:
            rows.append({"zona": "fondo", "rx": float(rx_bg), "geometry": bg})
    gdf = gpd.GeoDataFrame(rows, columns=["zona", "rx", "geometry"], geometry="geometry", crs=crs)
    gdf["unidad"] = unit
    gdf["ha"] = gdf.geometry.area / 1e4
    return gdf[COLS]


def summary(rx, boundary_area_m2):
    """Hectáreas por zona, % del lote y producto total."""
    out = {"lote_ha": boundary_area_m2 / 1e4}
    for zona in ("maleza", "fondo"):
        sel = rx[rx["zona"] == zona]
        out[zona + "_ha"] = float(sel["ha"].sum())
        out[zona + "_producto"] = float((sel["ha"] * sel["rx"]).sum())
    return out


# ---------- grilla común a ISOXML y GeoTIFF ----------

def rate_grid(rx, boundary, cell_m):
    """Grilla en WGS84 (celdas de ~cell_m) con el código de zona: 0 fuera del lote, 1 resto del lote (fondo o sin
    aplicar), 2 maleza. La maleza se rasteriza con all_touched: toda celda que toca una zona se aplica."""
    b4326 = gpd.GeoSeries([boundary], crs=rx.crs).to_crs("EPSG:4326").iloc[0]
    minx, miny, maxx, maxy = b4326.bounds
    lat = (miny + maxy) / 2
    dlat = cell_m / 111_320.0
    dlon = cell_m / (111_320.0 * np.cos(np.radians(lat)))
    ncols, nrows = int(np.ceil((maxx - minx) / dlon)), int(np.ceil((maxy - miny) / dlat))
    transform = from_origin(minx, miny + nrows * dlat, dlon, dlat)  # fila 0 = norte (como un raster)
    codes = rasterize([(b4326, 1)], out_shape=(nrows, ncols), transform=transform, fill=0, dtype="uint8")
    weed = rx[rx["zona"] == "maleza"].to_crs("EPSG:4326")
    if len(weed):
        w = rasterize([(g, 1) for g in weed.geometry], out_shape=(nrows, ncols), transform=transform, fill=0,
                      all_touched=True, dtype="uint8")
        codes[(w == 1) & (codes > 0)] = 2
    return codes, transform, (minx, miny, dlon, dlat)


# ---------- exports ----------

def _zip(paths, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, os.path.basename(p))


def export_shapefile(rx, out_dir, name):
    """Una fila por zona (multipolígono: maleza y, si hay, fondo) en WGS84 con rx (dosis), unidad, zona y ha.
    Nombres de campo <= 10 caracteres (DBF)."""
    os.makedirs(out_dir, exist_ok=True)
    gdf = rx.to_crs("EPSG:4326")  # build_prescription ya deja una fila por zona y el área en ha
    shp = os.path.join(out_dir, f"{name}.shp")
    gdf.to_file(shp, driver="ESRI Shapefile")
    parts = [os.path.join(out_dir, f"{name}{ext}") for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg")]
    _zip([p for p in parts if os.path.exists(p)], os.path.join(out_dir, f"{name}.zip"))
    return shp


def export_geotiff(rx, boundary, path, cell_m, nodata=-1.0):
    """Mapa de dosis float32 en WGS84; fuera del lote = nodata."""
    codes, transform, _ = rate_grid(rx, boundary, cell_m)
    rates = {z: float(rx.loc[rx["zona"] == z, "rx"].iloc[0]) if (rx["zona"] == z).any() else 0.0
             for z in ("maleza", "fondo")}
    data = np.full(codes.shape, nodata, np.float32)
    data[codes == 1] = rates["fondo"]
    data[codes == 2] = rates["maleza"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=codes.shape[1], height=codes.shape[0], count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform, nodata=nodata,
                       compress="deflate") as dst:
        dst.write(data, 1)
    return path


def export_dji(rx, boundary, out_dir, name, cell_m):
    """Carpeta DJI/ para la microSD del control de Agras: DJI/Shapefile (límite) y DJI/Rx (dosis en GeoTIFF).
    En el control: Prescription Map > Map Source 'Other', Source unit 'ha'."""
    root = os.path.join(out_dir, "DJI")
    lim = gpd.GeoDataFrame({"nombre": [name]}, geometry=[boundary], crs=rx.crs).to_crs("EPSG:4326")
    os.makedirs(os.path.join(root, "Shapefile"), exist_ok=True)
    lim.to_file(os.path.join(root, "Shapefile", f"{name}.shp"), driver="ESRI Shapefile")
    export_geotiff(rx, boundary, os.path.join(root, "Rx", f"{name}.tif"), cell_m, nodata=0.0)
    return root


def export_isoxml(rx, boundary, out_dir, name, cell_m, unit="L/ha", product="Herbicida"):
    """TASKDATA/TASKDATA.XML (ISOXML v3) con una tarea planificada y grilla tipo 1 (un byte por celda = código de
    zona de tratamiento). Zonas: 0 fuera del lote (dosis 0), 1 fondo / sin aplicar, 2 maleza."""
    ddi, factor = UNITS[unit]
    codes, _, (minx, miny, dlon, dlat) = rate_grid(rx, boundary, cell_m)
    rates = {z: float(rx.loc[rx["zona"] == z, "rx"].iloc[0]) if (rx["zona"] == z).any() else 0.0
             for z in ("maleza", "fondo")}
    task_dir = os.path.join(out_dir, "TASKDATA")
    os.makedirs(task_dir, exist_ok=True)
    # ISOXML: la primera celda es la esquina sudoeste; filas hacia el norte y columnas hacia el este
    np.ascontiguousarray(codes[::-1]).tofile(os.path.join(task_dir, "GRD00001.BIN"))

    def pdv(value):
        return f'<PDV A="{ddi:04X}" B="{int(round(value * factor))}" C="PDT1"/>'

    b4326 = gpd.GeoSeries([boundary], crs=rx.crs).to_crs("EPSG:4326").iloc[0]
    polys = list(getattr(b4326, "geoms", [b4326]))
    pln = []
    for poly in polys:  # límite del lote: PLN tipo 1 (borde) con anillo exterior (LSG 1) e interiores (LSG 2)
        rings = [(1, poly.exterior)] + [(2, r) for r in poly.interiors]
        lsg = "".join(f'<LSG A="{t}">' + "".join(f'<PNT A="2" C="{y:.9f}" D="{x:.9f}"/>' for x, y in r.coords)
                      + "</LSG>" for t, r in rings)
        pln.append(f'<PLN A="1">{lsg}</PLN>')
    area = int(round(gpd.GeoSeries([boundary], crs=rx.crs).area.iloc[0]))
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<ISO11783_TaskData VersionMajor="3" VersionMinor="3" ManagementSoftwareManufacturer="deteccion-maleza" '
        'ManagementSoftwareVersion="1.0" DataTransferOrigin="1">\n'
        f'<PDT A="PDT1" B="{product}"/>\n'
        f'<PFD A="PFD1" C="{name}" D="{area}">{"".join(pln)}</PFD>\n'
        f'<TSK A="TSK1" B="{name} {datetime.now():%Y-%m-%d}" E="PFD1" G="1" H="1" I="1" J="0">'
        f'<TZN A="0" B="Fuera del lote">{pdv(0)}</TZN>'
        f'<TZN A="1" B="Fondo">{pdv(rates["fondo"])}</TZN>'
        f'<TZN A="2" B="Maleza">{pdv(rates["maleza"])}</TZN>'
        f'<GRD A="{miny:.9f}" B="{minx:.9f}" C="{dlat:.12f}" D="{dlon:.12f}" E="{codes.shape[1]}" '
        f'F="{codes.shape[0]}" G="GRD00001" I="1"/>'
        '</TSK>\n'
        '</ISO11783_TaskData>\n')
    with open(os.path.join(task_dir, "TASKDATA.XML"), "w", encoding="utf-8") as f:
        f.write(xml)
    return task_dir


FORMATS = {
    "shp": "Shapefile (John Deere, Trimble, Raven, Ag Leader y monitores en general)",
    "isoxml": "ISOXML (terminales ISOBUS: Case, New Holland, AGCO, CLAAS, Amazone…)",
    "dji": "DJI Agras (límite SHP + mapa de dosis TIFF)",
}


def export_prescription(rx, boundary, out_dir, name, formats, cell_m, unit="L/ha"):
    """Exporta en los formatos pedidos; devuelve {formato: ruta}."""
    out = {}
    if "shp" in formats:
        out["shp"] = export_shapefile(rx, os.path.join(out_dir, "shapefile"), name)
    if "isoxml" in formats:
        out["isoxml"] = export_isoxml(rx, boundary, out_dir, name, cell_m, unit)
    if "dji" in formats:
        out["dji"] = export_dji(rx, boundary, out_dir, name, cell_m)
    return out


def safe_name(text):
    """Nombre de archivo corto y sin caracteres raros (algunos monitores no leen acentos ni espacios)."""
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(text).stem)
    return keep.encode("ascii", "ignore").decode()[:40] or "prescripcion"
