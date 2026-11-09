"""Mapa de prescripción a partir de los polígonos de maleza y export por monitor de pulverizadora o dron (sin UI).

Zonas: "maleza" (polígonos con buffer +pos, disueltos y buffer -neg) y, opcional, "fondo" (el resto del lote).
La geometría base es una sola; cada monitor (PROFILES) define formato, carpeta del pendrive, máximo de polígonos,
si admite huecos y si se carga como límite. El fondo va solo si se pide. Referencia y fuentes: docs/prescripciones_monitores.md.
Ningún perfil se probó todavía en un monitor real.
"""
import math
import os
import struct
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import datetime

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from shapely import make_valid, set_precision
from shapely.geometry import box, shape
from shapely.ops import unary_union

# unidad -> (DDI ISOBUS de dosis por área, factor de la unidad a la del DDI)
# DDI 1: Setpoint Volume Per Area Application Rate (mm³/m²; 1 L/ha = 100). DDI 6: Setpoint Mass Per Area (mg/m²;
# 1 kg/ha = 100).
UNITS = {"L/ha": (1, 100.0), "kg/ha": (6, 100.0)}
COLS = ["zona", "rx", "unidad", "ha", "geometry"]
RATE_FIELD = "RATE"  # único campo del shapefile; se usó a mano en el caso de IntelliView IV (ver el doc)
SIMPLIFY_M = 0.2  # en lote26 cambió el área menos de 1%
MIN_PIECE_M2 = 0.01  # astillas que dejan la diferencia y los cortes


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


# ---------- perfiles por monitor (docs/prescripciones_monitores.md) ----------

@dataclass(frozen=True)
class Monitor:
    label: str
    format: str | None  # shp, isoxml, dji; None = la app no puede generar su archivo
    folder: str = ""  # carpeta del shapefile desde la raíz del pendrive; "" = raíz
    max_polys: int | None = None
    holes: bool = True  # False: se parten los anillos de la zona de maleza
    as_boundary: bool = False  # True: archivo para cargar como límite (solo la maleza, un registro; ver export_shapefile)
    note: str = ""


_PIQ = ("En el monitor (PiQ ≥ 14.60): ajuste que crea zonas de exclusión desde las zonas a 0, o «Turn Off All Sections "
        "on Zero Target Rate» (corta todo el botalón); para eso marcar «Agregar fondo» con dosis 0. Activar «Enable "
        "Prescription Usage Automatically». Requiere licencia de prescripciones.")
_AGCO = "El monitor espera un solo TASKDATA con todos los lotes; la app genera uno por lote."
_NO_FOLDER = "Carpeta sin verificar: se exporta en la raíz; revisar la guía del fabricante."
_NO_DOCS = "Formato y carpeta sin documentación pública."

PROFILES = {
    "ag_leader": Monitor("Ag Leader InCommand 800/1200, Integra", "shp",
                         note="Reporte sin verificar: no corta secciones en manchones menores al ancho del botalón."),
    "agco": Monitor("AGCO C2000/C3000, Massey DataTronic 5, Valtra SmartTouch", "isoxml", note=_AGCO),
    "arag": Monitor("ARAG Bravo 400S / Delta 80", "shp", "maps"),
    "case_pro1200": Monitor("Case IH Pro 1200 / NH IntelliView 12", "shp"),
    "case_pro700": Monitor("Case IH Pro 700 / NH IntelliView III-IV", "shp", "Shapefile", max_polys=1000,
                           as_boundary=True,
                           note="Cargar como límite: copia el formato de un archivo de Avansys que funcionó así en una "
                                "IntelliView IV. Solo lleva la zona de maleza (sin fondo) y rx = 0; la dosis no va en "
                                "el archivo."),
    "cuffia_ag9500": Monitor("Abelardo Cuffia AG9500", "shp", max_polys=50, note=_NO_FOLDER),
    "cuffia_fusion": Monitor("Abelardo Cuffia AG Fusion", "shp", max_polys=100, note=_NO_FOLDER),
    "cuffia_fusionpro": Monitor("Abelardo Cuffia AG Fusion Pro", "shp", max_polys=100,
                                note=_NO_FOLDER + " Máximo de polígonos sin verificar: se usa el de AG Fusion."),
    "deere_gen4": Monitor("John Deere GS 2630 / Gen 4 (4640, 4600) / G5", "shp", "Rx",
                          note="Con Section Control, la dosis 0 apaga secciones (documentado por Deere). "
                               "En Rx/ pueden ir todos los lotes juntos."),
    "deere_2600": Monitor("John Deere GS 2600", "shp"),
    "hexagon": Monitor("Hexagon Ti5/Ti7/Ti10", "shp"),
    "jacto": Monitor("Jacto Omni 700", "shp"),
    "microelec": Monitor("Microelec (fertilización/pulverización)", "shp", note=_NO_FOLDER),
    "muller": Monitor("Müller Touch 800/1200", "shp", "SHP", note="Requiere activar la licencia."),
    "plantium": Monitor("Plantium SBOX 7/11", "shp", note=f"Al cargarlo, elegir la columna de dosis {RATE_FIELD}."),
    "raven_viper": Monitor("Raven Viper 4/4+ / Viper Pro", "shp", "Raven/Misc/rxMaps",
                           note="Viper Pro: las zonas a 0 no cortan solas, hay que configurarlas como «no-spray zones»."),
    "raven_envizio": Monitor("Raven Envizio Pro", "shp", note=_NO_FOLDER),
    "stara": Monitor("Stara Topper 4500/5500/6500", "shp", "Dados/Mapas",
                     note="También se carga por Telemetría Stara."),
    "teejet": Monitor("TeeJet Matrix Pro GS / Aeros", "isoxml",
                      note="Importar TASKDATA en Fieldware Link (en la PC) y exportar el trabajo a la consola."),
    "topcon": Monitor("Topcon X25/X30/X35", "shp", "VRC"),
    "trimble_piq": Monitor("Trimble GFX-350/750/1060/1260, TMX-2050 (Precision IQ)", "shp", "AgData/Prescriptions",
                           holes=False, note=_PIQ),
    "trimble_fmx": Monitor("Trimble FmX / CFX-750", "shp", "AgGPS/Prescriptions", holes=False,
                           note="CFX-750 necesita el desbloqueo de dosis variable. FmX con factor de escala 1."),
    "dji_agras": Monitor("DJI Agras (dron)", "dji",
                         note="En el control: Prescription Map > Map Source «Other», Source unit «ha»."),
    "shp_generico": Monitor("Otro monitor: Shapefile genérico", "shp",
                            note="Sin carpeta ni ajustes de un monitor en particular."),
    "isoxml_generico": Monitor("Otro monitor: ISOXML genérico (TASKDATA)", "isoxml"),
    # sin archivo propio: figuran deshabilitados para que se vea por qué no se pueden elegir
    "controlagro": Monitor("ControlAgro CAS 4500 (requiere conversor)", None,
                           note="Pide .SFV convertido con el software MHS."),
    "tim": Monitor("TIM CS Oro Máx / TS Oro (requiere conversor)", None, note="Pide .srm convertido con proMAP."),
    "verion": Monitor("Verion Vcom 7/10 (requiere conversor)", None, note="Pide un archivo convertido con Agromapas 4.0."),
    "case_pro600": Monitor("Case IH Pro 600 (solo .cn1)", None, note="Solo lee el formato .cn1."),
    "caiman": Monitor("Caimán (sin verificar)", None, note=_NO_DOCS),
    "metalfor": Monitor("Metalfor (sin verificar)", None, note=_NO_DOCS),
    "sensor": Monitor("Sensor (sin verificar)", None, note=_NO_DOCS),
}
DEFAULT_MONITOR = "shp_generico"


def min_patch_m2(section_m, speed_kmh, delay_s):
    """Superficie mínima de un manchón: ancho de sección x distancia recorrida durante el retardo de las válvulas."""
    return section_m * speed_kmh / 3.6 * delay_s


def _parts(g):
    return [p for p in getattr(g, "geoms", [g]) if not p.is_empty]


def _grow_small_patches(weed, min_m2):
    """Agranda los manchones chicos hasta ~min_m2 en vez de descartarlos: no se pierde maleza. Los que se tocan se unen."""
    return unary_union([p if p.area >= min_m2 else p.buffer(math.sqrt(min_m2 / math.pi) - math.sqrt(p.area / math.pi))
                        for p in _parts(weed)])


def _to_grid(weed, boundary, cell_m):
    """Maleza en bloques cuadrados de cell_m, alineados al norte y al CRS: toda celda que toca maleza se aplica entera."""
    minx, miny, maxx, maxy = boundary.bounds
    transform = from_origin(minx, maxy, cell_m, cell_m)
    size = (math.ceil((maxy - miny) / cell_m), math.ceil((maxx - minx) / cell_m))
    cells = rasterize(_parts(weed), out_shape=size, transform=transform, all_touched=True, dtype="uint8")
    return unary_union([make_valid(shape(g)) for g, _ in shapes(cells, mask=cells > 0, transform=transform)])


def _split_holes(poly):
    """Polígonos sin huecos que cubren lo mismo: corta en vertical por el hueco de la mediana, hasta que no queden."""
    if not poly.interiors:
        return [poly]
    x = sorted(r.centroid.x for r in poly.interiors)[len(poly.interiors) // 2]
    minx, miny, maxx, maxy = poly.bounds
    out = []
    for half in (box(minx - 1, miny - 1, x, maxy + 1), box(x, miny - 1, maxx + 1, maxy + 1)):
        for part in _parts(_polygonal(poly.intersection(half))):
            out.extend(_split_holes(part))
    return out


def adapt_to_monitor(rx, boundary, key, unit, min_patch=0.0, grid_m=0.0):
    """Zonas y límite como los necesita el monitor. Devuelve (rx, boundary) en el mismo CRS métrico.
    grid_m > 0: la maleza se pasa a bloques cuadrados de ese lado, después de agrandar los manchones chicos.

    Shapefile: simplificado y una fila por polígono simple; maleza sin anillos si el perfil lo pide. El fondo va solo si
    se pidió y se recalcula como lote - maleza, así las zonas no se pisan ni dejan rendijas después de agrandar y
    simplificar. Perfil de límite (as_boundary): solo la maleza, en una fila multiparte."""
    monitor = PROFILES[key]
    is_shp = monitor.format == "shp"
    weed_rows, bg_rows = rx[rx["zona"] == "maleza"], rx[rx["zona"] == "fondo"]
    weed = unary_union(list(weed_rows.geometry)) if len(weed_rows) else None
    if is_shp:
        boundary = _polygonal(make_valid(boundary.simplify(SIMPLIFY_M)))
    if weed is not None:
        if min_patch > 0:
            weed = _grow_small_patches(weed, min_patch)
        if grid_m > 0:
            weed = _to_grid(weed, boundary, grid_m)
        if is_shp:
            weed = make_valid(weed.simplify(SIMPLIFY_M))
        weed = _polygonal(weed.intersection(boundary))
    zones = []
    if weed is not None and not weed.is_empty:
        zones.append(("maleza", float(weed_rows["rx"].iloc[0]), weed))
    if len(bg_rows) and not monitor.as_boundary:
        bg = _polygonal(boundary.difference(weed)) if weed is not None else boundary
        if not bg.is_empty:
            zones.append(("fondo", float(bg_rows["rx"].iloc[0]), bg))
    rows = []
    for zona, rate, geom in zones:
        pieces = [geom]
        if is_shp:
            pieces = _parts(geom)
            if not monitor.holes and zona == "maleza":  # el fondo rodea la maleza: partirlo solo sumaría polígonos
                pieces = [q for p in pieces for q in _split_holes(p)]
            pieces = [p for p in pieces if p.area >= MIN_PIECE_M2]
            if monitor.as_boundary and pieces:
                pieces = [unary_union(pieces)]
        rows.extend({"zona": zona, "rx": rate, "geometry": p} for p in pieces)
    out = gpd.GeoDataFrame(rows, columns=["zona", "rx", "geometry"], geometry="geometry", crs=rx.crs)
    out["unidad"] = unit
    out["ha"] = out.geometry.area / 1e4
    return out[COLS], boundary


def n_polygons(rx):
    """Polígonos simples: en los perfiles de límite una fila tiene muchas partes."""
    return sum(len(_parts(g)) for g in rx.geometry)


def over_limit(rx, key):
    """Máximo de polígonos del monitor si esta prescripción lo supera; si no, None."""
    limit = PROFILES[key].max_polys
    return limit if limit is not None and n_polygons(rx) > limit else None


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

def _zip_tree(root, zip_path):
    """Zip cuyo contenido es lo que hay dentro de root: se descomprime directo en la raíz del pendrive."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, _, files in os.walk(root):
            for f in files:
                full = os.path.join(folder, f)
                z.write(full, os.path.relpath(full, root))


def _write_dbf(path, field, width, decimals, values):
    """DBF con un solo campo numérico; values ya formateados (texto de hasta width caracteres). pyogrio no deja elegir
    el ancho y escribe N 24.15."""
    today = datetime.now()
    with open(path, "wb") as f:
        f.write(struct.pack("<BBBBIHH20x", 3, today.year - 1900, today.month, today.day, len(values), 65, width + 1))
        f.write(struct.pack("<11scxxxxBB14x", field.encode("ascii"), b"N", width, decimals) + b"\r")
        for v in values:
            f.write(b" " + v.rjust(width).encode("ascii"))
        f.write(b"\x1a")


# .prj de WGS84 en WKT de ESRI, el que traen los shapefiles de Avansys que lee la IntelliView IV
WGS84_PRJ = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
             'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')


def _valid_wgs84(rx):
    """Pasa a WGS84 con polígonos simples y válidos. Un hueco que toca un borde es válido en metros, pero al reproyectar
    puede cruzarlo por una fracción de milímetro; redondear a 1e-8° (~1 mm) lo resuelve."""
    gdf = rx.to_crs("EPSG:4326")
    gdf = gdf.set_geometry(set_precision(make_valid(np.asarray(gdf.geometry.values)), 1e-8)).explode(index_parts=False)
    return gdf[(gdf.geom_type == "Polygon") & ~gdf.is_empty]


def export_shapefile(rx, out_dir, name, as_boundary=False):
    """Una fila por polígono en WGS84 (con .prj), con el campo de dosis como único atributo.

    as_boundary: copia del shapefile de Avansys que funciona cargado como límite en una IntelliView IV (ver el doc):
    toda la maleza en un solo registro multiparte (con huecos), campo rx N 5.4 con valor 0 y .cpg ISO-8859-1."""
    os.makedirs(out_dir, exist_ok=True)
    gdf = _valid_wgs84(rx)
    if as_boundary:
        gdf = gpd.GeoDataFrame({"rx": [0.0]}, geometry=[unary_union(gdf.geometry.values)], crs=gdf.crs)
    shp = os.path.join(out_dir, f"{name}.shp")
    gdf[["rx", "geometry"]].rename(columns={"rx": RATE_FIELD}).to_file(shp, driver="ESRI Shapefile")
    dbf = os.path.join(out_dir, f"{name}.dbf")
    if as_boundary:
        _write_dbf(dbf, "rx", 5, 4, ["0.000"])  # mismos bytes que el archivo de Avansys
    else:
        _write_dbf(dbf, RATE_FIELD, 10, 2, [f"{r:.2f}" for r in gdf["rx"]])
    with open(os.path.join(out_dir, f"{name}.prj"), "w", encoding="ascii") as f:
        f.write(WGS84_PRJ)
    cpg = os.path.join(out_dir, f"{name}.cpg")  # sin campos de texto no hace falta, salvo para copiar a Avansys
    if as_boundary:
        with open(cpg, "w", encoding="ascii") as f:
            f.write("ISO-8859-1")
    elif os.path.exists(cpg):
        os.remove(cpg)
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


def export_for_monitor(rx, boundary, out_dir, name, key, cell_m, unit="L/ha"):
    """Arma <out_dir>/<name>_<monitor>/ con la estructura del pendrive y el mismo contenido en un .zip al lado.
    rx ya tiene que venir de adapt_to_monitor. Devuelve (carpeta, zip)."""
    monitor = PROFILES[key]
    root = os.path.join(out_dir, f"{name}_{key}")
    if monitor.format == "shp":
        shp_name = f"{datetime.now():%Y%m%d}-{name}" if monitor.as_boundary else name  # como Avansys: 20261108-LOTE_28
        export_shapefile(rx, os.path.join(root, *monitor.folder.split("/")), shp_name, monitor.as_boundary)
    elif monitor.format == "isoxml":
        export_isoxml(rx, boundary, root, name, cell_m, unit)
    elif monitor.format == "dji":
        export_dji(rx, boundary, root, name, cell_m)
    else:
        raise ValueError(f"La app no genera archivos para {monitor.label}.")
    zip_path = root + ".zip"
    _zip_tree(root, zip_path)
    return root, zip_path


def safe_name(text):
    """Nombre de archivo corto y sin caracteres raros (algunos monitores no leen acentos ni espacios)."""
    plain = unicodedata.normalize("NFKD", text.strip()).encode("ascii", "ignore").decode()  # Ñandú -> Nandu
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in plain)
    return keep[:40] or "prescripcion"
