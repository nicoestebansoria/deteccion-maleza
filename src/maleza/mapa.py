"""Mapa folium y datos del ortomosaico para la app (sin UI)."""
import glob
import html
import io
import os

import folium
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

PREVIEW_MAX_PX = 2048
LOT_LABEL_STYLE = ("transform:translate(-50%,-50%);width:max-content;background:rgba(0,0,0,.6);color:#fff;"
                   "padding:1px 6px;border-radius:3px;font:12px sans-serif")


def ortho_preview(path):
    """Imagen reducida del ortomosaico (WebP con transparencia) + bounds en WGS84 para superponer en el mapa.

    Leaflet estira el ImageOverlay linealmente en Web Mercator: si la imagen quedara en su CRS
    original (ej. UTM) se dibujaría girada/corrida (en el orto de prueba, hasta 12 m en las
    esquinas). Por eso se reproyecta a EPSG:3857 antes.
    """
    with rasterio.open(path) as src, WarpedVRT(src, crs="EPSG:3857", resampling=Resampling.bilinear) as vrt:
        scale = max(vrt.width, vrt.height) / PREVIEW_MAX_PX
        out_w, out_h = int(vrt.width / max(scale, 1)), int(vrt.height / max(scale, 1))
        rgb = vrt.read([1, 2, 3], out_shape=(3, out_h, out_w), resampling=Resampling.average)
        rgb = np.moveaxis(rgb, 0, -1)
        if rgb.dtype != np.uint8:  # ortos de 16 bits: estirar al rango 2-98%
            lo, hi = np.percentile(rgb[rgb > 0], [2, 98]) if (rgb > 0).any() else (0, 1)
            rgb = (np.clip((rgb - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
        # máscara de datos válidos (banda alfa / nodata del raster + áreas fuera tras reproyectar)
        alpha = vrt.dataset_mask(out_shape=(out_h, out_w), resampling=Resampling.nearest).astype(np.uint8)
        buf = io.BytesIO()
        # WebP: ~10 veces más liviano que PNG; va embebido en el HTML del mapa en cada redibujo
        Image.fromarray(np.dstack([rgb, alpha])).save(buf, format="WEBP", quality=85)
        # en 3857 el rectángulo está alineado con lon/lat: estos bounds son exactos
        bounds = transform_bounds("EPSG:3857", "EPSG:4326", *vrt.bounds)
    return buf.getvalue(), bounds


def ortho_info(path):
    with rasterio.open(path) as src:
        return {"ancho": src.width, "alto": src.height, "bandas": src.count, "dtype": src.dtypes[0],
                "crs": str(src.crs), "res": src.res, "geografico": bool(src.crs and src.crs.is_geographic)}


def list_orthos(cfg):
    found = []
    for d in ("orthos", "uploads"):
        for ext in ("tif", "tiff"):
            found += glob.glob(os.path.join(cfg["root"], "data", d, f"*.{ext}"))
    return sorted(found)


def available_weights(cfg):
    runs = cfg["paths"]["runs_dir"]
    found = sorted(glob.glob(os.path.join(runs, "**", "weights", "best.pt"), recursive=True))
    found += sorted(glob.glob(os.path.join(runs, "**", "*_openvino_model"), recursive=True))
    default = cfg["predict"]["weights"]
    if default and os.path.exists(default):  # el de config.yaml va primero: es el que la app elige al abrir
        found = [default] + [f for f in found if os.path.normcase(f) != os.path.normcase(default)]
    return found


def build_map(preview_url, bounds, boxes=None, polys=None, view=None, boxes_visible=False, tiles=None,
              tiles_min_zoom=18, perimeter=None, rx=None, lots=None):
    """preview_url: URL (o data URI) de la imagen de ortho_preview; bounds: (w, s, e, n) en WGS84.
    view: (lat, lon, zoom) para conservar el encuadre al redibujar; sin view se encuadra el orto.
    tiles: (url XYZ, zoom nativo) de tiles.TileServer: desde tiles_min_zoom el orto se ve a resolución completa.
    perimeter: GeoDataFrame del perímetro del lote (se dibuja solo el contorno).
    rx: GeoDataFrame de prescripción.build_prescription (zonas maleza / fondo).
    lots: GeoDataFrame de lotes para elegir (columnas nombre, elegido): contorno con su nombre, celeste si está elegido."""
    w, s, e, n = bounds
    center, zoom = ([view[0], view[1]], view[2]) if view else ([(s + n) / 2, (w + e) / 2], 17)
    m = folium.Map(location=center, zoom_start=zoom, max_zoom=24, tiles=None)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satélite", max_zoom=24,
        max_native_zoom=17,  # en zonas rurales Esri no tiene imagen más fina: de cerca muestra "Map data not available"
    ).add_to(m)
    # panel propio para el orto: arriba del satélite y debajo de los polígonos (overlayPane, z 400)
    folium.map.CustomPane("orto", z_index=350).add_to(m)
    orto = folium.FeatureGroup(name="Ortomosaico").add_to(m)
    folium.raster_layers.ImageOverlay(image=preview_url, bounds=[[s, w], [n, e]], pane="orto").add_to(orto)
    if tiles:
        url, native_zoom = tiles
        folium.TileLayer(url, attr="Ortomosaico", pane="orto", min_zoom=tiles_min_zoom, max_zoom=24,
                         max_native_zoom=native_zoom, bounds=[[s, w], [n, e]], overlay=True).add_to(orto)

    if lots is not None and len(lots):
        lots = lots[["nombre", "elegido", "geometry"]].to_crs("EPSG:4326")
        group = folium.FeatureGroup(name="Lotes").add_to(m)
        folium.GeoJson(lots, style_function=lambda f: {"color": "#00e5ff", "weight": 2.5, "fill": False}
                       if f["properties"]["elegido"] else {"color": "#ffffff", "weight": 1.5, "dashArray": "6 6",
                                                           "fill": False},
                       interactive=False).add_to(group)
        for nombre, geom in zip(lots["nombre"], lots.geometry):  # un tooltip permanente solo se abre en el primer lote
            at = geom.representative_point()
            folium.Marker([at.y, at.x], interactive=False, icon=folium.DivIcon(
                icon_size=(0, 0), html=f'<div style="{LOT_LABEL_STYLE}">{html.escape(nombre)}</div>')).add_to(group)
    if rx is not None and len(rx):
        folium.GeoJson(
            rx.to_crs("EPSG:4326"), name="Prescripción",
            style_function=lambda f: {"color": "#b000ff" if f["properties"]["zona"] == "maleza" else "#2e7d32",
                                      "weight": 1.5, "fillOpacity": 0.3 if f["properties"]["zona"] == "maleza" else 0.15},
            tooltip=folium.GeoJsonTooltip(["zona", "rx", "unidad", "ha"], aliases=["Zona", "rx", "Unidad", "ha"],
                                          localize=True),
        ).add_to(m)
    if perimeter is not None:
        folium.GeoJson(perimeter[["nombre", "geometry"]].to_crs("EPSG:4326"), name="Perímetro del lote",
                       style_function=lambda f: {"color": "#00e5ff", "weight": 2.5, "fill": False},
                       interactive=False).add_to(m)
    if polys is not None and len(polys):
        folium.GeoJson(
            polys.to_crs("EPSG:4326"), name="Malezas (polígonos)",
            style_function=lambda f: {"color": "#ff8c00" if f["properties"]["source"] == "exg" else "#ff2d2d",
                                      "weight": 1.5, "fillOpacity": 0.35},
            tooltip=folium.GeoJsonTooltip(
                [c for c in ("confidence", "area_m2", "source", "metodo", "verde_pico") if c in polys],
                aliases=[a for c, a in [("confidence", "Confianza"), ("area_m2", "Área m²"), ("source", "Origen"),
                                        ("metodo", "Método"), ("verde_pico", "Verde pico")] if c in polys],
                localize=True),
        ).add_to(m)
    if boxes is not None and len(boxes):
        folium.GeoJson(
            boxes[["confidence", "source", "geometry"]].to_crs("EPSG:4326"), name="Detecciones (cajas)",
            show=boxes_visible,
            style_function=lambda f: {"color": "#00b0ff" if f["properties"]["source"] == "manual" else "#ffd400",
                                      "weight": 1.2, "fillOpacity": 0},
            tooltip=folium.GeoJsonTooltip(["confidence", "source"], aliases=["Confianza", "Origen"], localize=True),
        ).add_to(m)
    folium.LayerControl().add_to(m)
    if not view:
        m.fit_bounds([[s, w], [n, e]])
    return m
