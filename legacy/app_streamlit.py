"""App web local: cargar un ortomosaico, detectar malezas (modelo + complemento por índice de verde),
ver los polígonos, descargarlos y reentrenar.

Lanzar con:  uv run maleza app
"""
import base64
import glob
import hashlib
import io
import math
import os
import zipfile
from pathlib import Path

import folium
import numpy as np
import rasterio
import streamlit as st
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from shapely.geometry import box as shapely_box
from streamlit_folium import st_folium

from maleza import review as rv
from maleza.common import load_config
from maleza.polygonize import export, green_components, polygonize_green
from maleza.predict import detect_ortho, load_model
from maleza.train import launch_training, run_status

CONFIG_PATH = os.environ.get("MALEZA_CONFIG", "config.yaml")
PREVIEW_MAX_PX = 2048

st.set_page_config(page_title="Detección de malezas", layout="wide")

# tarjetas de métricas: borde fino + barra de acento a la izquierda (colores relativos al tema)
st.markdown("""<style>
.cards {display: grid; grid-template-columns: repeat(var(--cols), minmax(0, 1fr)); gap: 1.2rem; margin: .4rem 0 1.2rem;}
.card {border: 1px solid rgba(128,128,128,.35); border-left: 7px solid currentColor; border-radius: 6px;
       padding: 1.1rem 1.4rem;}
.card .lbl {font-size: .95rem; opacity: .85;}
.card .val {font-size: 2.1rem; line-height: 1.25; margin-top: .3rem; overflow-wrap: anywhere;}
.cards-title {font-weight: 600; font-size: 1.05rem; margin-top: .6rem;}
</style>""", unsafe_allow_html=True)


def cards(items, title=None, cols=4):
    """items: [(etiqueta, valor) o (etiqueta, valor, tooltip)]."""
    import html
    body = "".join(
        f'<div class="card" title="{html.escape(it[2]) if len(it) > 2 else ""}">'
        f'<div class="lbl">{html.escape(str(it[0]))}</div><div class="val">{html.escape(str(it[1]))}</div></div>'
        for it in items)
    head = f'<div class="cards-title">{html.escape(title)}</div>' if title else ""
    st.markdown(f'{head}<div class="cards" style="--cols:{cols}">{body}</div>', unsafe_allow_html=True)


@st.cache_resource
def get_config():
    return load_config(CONFIG_PATH)


@st.cache_resource
def get_model(weights):
    return load_model(weights)


@st.cache_data
def ortho_preview(path):
    """PNG reducido del ortomosaico + bounds en WGS84 para superponer en el mapa.

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
        Image.fromarray(np.dstack([rgb, alpha])).save(buf, format="PNG")
        # en 3857 el rectángulo está alineado con lon/lat: estos bounds son exactos
        bounds = transform_bounds("EPSG:3857", "EPSG:4326", *vrt.bounds)
    return buf.getvalue(), bounds


@st.cache_resource(max_entries=2, show_spinner="Calculando máscara de verde…")
def cached_green(path, threshold, resolution, close):
    cfg = get_config()
    pc = dict(cfg["polygonize"], exg_threshold=threshold, green_resolution_m=resolution, green_close_m=close)
    return green_components(path, dict(cfg, polygonize=pc))


def ortho_info(path):
    with rasterio.open(path) as src:
        return {"ancho": src.width, "alto": src.height, "bandas": src.count, "dtype": src.dtypes[0],
                "crs": str(src.crs), "res": src.res, "geografico": bool(src.crs and src.crs.is_geographic),
                "transform": src.transform}


def available_weights(cfg):
    runs = cfg["paths"]["runs_dir"]
    found = sorted(glob.glob(os.path.join(runs, "**", "weights", "best.pt"), recursive=True))
    found += sorted(glob.glob(os.path.join(runs, "**", "*_openvino_model"), recursive=True))
    default = cfg["predict"]["weights"]
    if default and os.path.exists(default) and default not in found:
        found.insert(0, default)
    return found


def zip_outputs(out_dir):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(Path(out_dir).iterdir()):
            if not f.name.startswith("revision"):
                z.write(f, f.name)
    return buf.getvalue()


def zoom_for_bounds(w, s, e, n, width_px=1100, height_px=600):
    """Zoom de Leaflet que encuadra los bounds (WGS84). Se calcula acá porque fit_bounds falla si el
    mapa se dibuja dentro de una pestaña oculta (Leaflet ve tamaño 0 y queda en zoom mínimo)."""
    def merc_y(lat):
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    dx = max(e - w, 1e-9) / 360
    dy = max(merc_y(n) - merc_y(s), 1e-9) / (2 * math.pi)
    z = min(math.log2(width_px / 256 / dx), math.log2(height_px / 256 / dy))
    return int(max(1, min(22, math.floor(z))))


def build_map(ortho_path, boxes, polys, reviewed_windows, transform, crs, focus=None, fit_layer=False,
              boxes_visible=True):
    """fit_layer: centrar y hacer zoom a los polígonos sin fit_bounds (para mapas en pestañas no visibles)."""
    png, (w, s, e, n) = ortho_preview(ortho_path)
    center, zoom = [(s + n) / 2, (w + e) / 2], 17
    if fit_layer and polys is not None and len(polys):
        lw, ls, le, ln = polys.to_crs("EPSG:4326").total_bounds
        center, zoom = [(ls + ln) / 2, (lw + le) / 2], zoom_for_bounds(lw, ls, le, ln)
    m = folium.Map(location=center, zoom_start=zoom, max_zoom=24, tiles=None)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satélite", max_zoom=24, max_native_zoom=19,
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        image="data:image/png;base64," + base64.b64encode(png).decode(),
        bounds=[[s, w], [n, e]], name="Ortomosaico",
    ).add_to(m)

    if reviewed_windows:
        import geopandas as gpd
        geoms = []
        for win in reviewed_windows:
            x1, y1 = transform * (win.col_off, win.row_off)
            x2, y2 = transform * (win.col_off + win.width, win.row_off + win.height)
            geoms.append(shapely_box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        folium.GeoJson(gpd.GeoDataFrame(geometry=geoms, crs=crs).to_crs("EPSG:4326"), name="Tiles revisados",
                       style_function=lambda f: {"color": "#00e676", "weight": 2, "fillOpacity": 0,
                                                 "dashArray": "6 4"}).add_to(m)
    if polys is not None and len(polys):
        folium.GeoJson(
            polys.to_crs("EPSG:4326"), name="Malezas (polígonos)",
            style_function=lambda f: {"color": "#ff8c00" if f["properties"]["source"] == "exg" else "#ff2d2d",
                                      "weight": 1.5, "fillOpacity": 0.35},
            tooltip=folium.GeoJsonTooltip(
                [c for c in ("confidence", "area_m2", "source", "metodo", "exg_pico") if c in polys],
                aliases=[a for c, a in [("confidence", "Confianza"), ("area_m2", "Área m²"), ("source", "Origen"),
                                        ("metodo", "Método"), ("exg_pico", "Verde pico")] if c in polys],
                localize=True),
        ).add_to(m)
    if boxes is not None and len(boxes):
        folium.GeoJson(
            boxes[["confidence", "source", "geometry"]].to_crs("EPSG:4326"), name="Detecciones (cajas)", show=boxes_visible,
            style_function=lambda f: {"color": "#00b0ff" if f["properties"]["source"] == "manual" else "#ffd400",
                                      "weight": 1.2, "fillOpacity": 0},
            tooltip=folium.GeoJsonTooltip(["confidence", "source"], aliases=["Confianza", "Origen"], localize=True),
        ).add_to(m)
    folium.LayerControl().add_to(m)
    if fit_layer:
        pass
    elif focus is not None:
        x1, y1 = transform * (focus.col_off, focus.row_off)
        x2, y2 = transform * (focus.col_off + focus.width, focus.row_off + focus.height)
        bw, bs, be, bn = transform_bounds(crs, "EPSG:4326", min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        m.fit_bounds([[bs, bw], [bn, be]])
    else:
        m.fit_bounds([[s, w], [n, e]])
    return m


def sidebar_training(cfg):
    st.header("Entrenamiento")
    pool = rv.training_pool_summary(cfg)
    st.caption("Tiles etiquetados en el pool (sin contar sintéticos):")
    st.dataframe({"ortomosaico": list(pool), "tiles": list(pool.values())}, hide_index=True,
                 width="stretch")

    runs = run_status(cfg["paths"]["runs_dir"])
    running = [r for r in runs if r["running"]]
    for r in runs[-6:]:
        estado = "🟢 entrenando" if r["running"] else "✔ terminado"
        map50 = f" · mAP50 {r['map50_best']:.3f}" if r["map50_best"] is not None else ""
        st.caption(f"**{r['name']}** — {estado} · época {r['epoch']}/{r['epochs']}{map50}")

    with st.expander("Reentrenar con todo el pool"):
        existing = {r["name"] for r in runs}
        n = 1
        while f"app_v{n}" in existing:
            n += 1
        name = st.text_input("Nombre de la corrida", f"app_v{n}")
        base = st.radio("Partir de", ["Modelo seleccionado (fine-tuning, más rápido)", "Pesos COCO (desde cero)"])
        epochs = st.number_input("Épocas", 10, 500, 60 if base.startswith("Modelo") else 150, step=10)
        st.caption("Arma un dataset nuevo en data/datasets/<nombre> (split espacial + sintéticos) y entrena en "
                   "segundo plano: sigue aunque cierres la app. En CPU ~3 min por época.")
        forzar = True
        if running:
            st.warning(f"Ya hay un entrenamiento en curso ({', '.join(r['name'] for r in running)}). "
                       "Dos a la vez compiten por la CPU y tardan el doble.")
            forzar = st.checkbox("Lanzar igual")
        if st.button("Lanzar entrenamiento", disabled=not forzar or name in existing):
            weights = st.session_state.get("weights") if base.startswith("Modelo") else None
            log = launch_training(CONFIG_PATH, name, weights=weights, epochs=int(epochs))
            st.success(f"Entrenamiento lanzado. Log: {log}")


# ---------------- UI ----------------

cfg = get_config()
st.title("Detección de malezas en ortomosaicos")

with st.sidebar:
    st.header("Modelo")
    weights_list = available_weights(cfg)
    if not weights_list:
        st.error(f"No encontré modelos en {cfg['paths']['runs_dir']}. Copiá tu best.pt de Drive a "
                 "runs/<nombre>/weights/best.pt o esperá a que el entrenamiento guarde el primero.")
    weights = st.selectbox("Pesos", weights_list, key="weights",
                           format_func=lambda p: os.path.relpath(p, cfg["paths"]["runs_dir"]))
    st.divider()
    sidebar_training(cfg)

if not weights_list:
    st.stop()

# --- 1. ortomosaico ---
st.subheader("1. Ortomosaico")
modo = st.radio("Origen", ["Carpeta del proyecto", "Subir archivo"], horizontal=True,
                help="Para ortomosaicos grandes (>1 GB) copialos a data/orthos: subirlos duplica el archivo.")
ortho_path = None
if modo == "Subir archivo":
    up = st.file_uploader("GeoTIFF RGB", type=["tif", "tiff"])
    if up is not None:
        upload_dir = os.path.join(cfg["root"], "data", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        ortho_path = os.path.join(upload_dir, up.name)
        if not os.path.exists(ortho_path) or os.path.getsize(ortho_path) != up.size:
            with open(ortho_path, "wb") as f:
                f.write(up.getbuffer())
else:
    candidatos = sorted(glob.glob(os.path.join(cfg["root"], "data", "orthos", "*.tif"))
                        + glob.glob(os.path.join(cfg["root"], "data", "uploads", "*.tif")))
    if candidatos:
        ortho_path = st.selectbox("Ortomosaicos en data/orthos y data/uploads", candidatos,
                                  format_func=os.path.basename)
    else:
        st.info("No hay ortomosaicos en data/orthos ni data/uploads: copiá ahí el .tif o usá 'Subir archivo'.")

if not ortho_path:
    st.stop()

info = ortho_info(ortho_path)
res_txt = f"{info['res'][0] * 100:.1f} cm/px" if not info["geografico"] else f"{info['res'][0]:.2e}°"
cards([("Tamaño", f"{info['ancho']} x {info['alto']} px"), ("Resolución", res_txt),
       ("Bandas", f"{info['bandas']} ({info['dtype']})"), ("CRS", info["crs"])], title="Ortomosaico")

train_res = cfg["predict"].get("train_resolution_m")
if info["geografico"]:
    st.warning("El CRS es geográfico (grados): las áreas en m² no van a ser correctas. "
               "Reproyectá a UTM (ej. EPSG:32720) antes.")
elif train_res and abs(info["res"][0] - train_res) / train_res > 0.3:
    st.warning(f"La resolución ({info['res'][0] * 100:.1f} cm/px) difiere bastante de la de "
               f"entrenamiento ({train_res * 100:.1f} cm/px): las plantas se ven de otro tamaño y "
               "la detección puede empeorar. Ideal: volar a la misma altura o remuestrear.")
if info["bandas"] < 3:
    st.error("Se necesita un ortomosaico RGB (3+ bandas).")
    st.stop()

# --- 2. detección ---
st.subheader("2. Detección")
det = st.session_state.get("det")
if det and (det["ortho"] != ortho_path or det["weights"] != weights):
    det = None
if st.button("Detectar malezas" if not det else "Volver a detectar", type="primary" if not det else "secondary"):
    model = get_model(weights)
    bar = st.progress(0.0, text="Detectando…")
    gdf, n_raw = detect_ortho(model, ortho_path, cfg, conf=cfg["predict"]["review_conf_min"],
                              progress=lambda d, t: bar.progress(d / t, text=f"Detectando… tile {d}/{t}"))
    bar.empty()
    det = {"ortho": ortho_path, "weights": weights, "raw": gdf, "n_raw": n_raw}
    st.session_state["det"] = det
    st.session_state.pop("polys", None)

if not det:
    st.stop()

# --- 3. resultado: modelo + complemento por verde (ExG) ---
st.subheader("3. Resultado")
raw = det["raw"]
pc = dict(cfg["polygonize"])
st.markdown("**Ajustes**")
conf_thr = st.slider("Confianza mínima del modelo", 0.05, 0.95, float(cfg["predict"]["conf"]), 0.05,
                     help="Filtra las detecciones en vivo sin volver a detectar.")
c1, c2 = st.columns(2)
pc["exg_threshold"] = c1.slider(
    "Umbral de verde (ExG)", 0.0, 0.2, float(pc["exg_threshold"]), 0.01,
    help="Más bajo = toma verdes más pálidos (y más rastrojo). Más alto = solo verde intenso.")
huerfanas = c2.checkbox("Agregar manchas verdes sin detección", value=True,
                        help="Manchas que el modelo no marcó: salen en naranja con origen 'exg' y "
                             "confianza 0. Incluye cualquier verde (también bordes del lote).")
pc["green_orphan_min_m2"] = c2.slider(
    "Área mínima de manchas sin detección (m²)", 0.05, 5.0, float(pc["green_orphan_min_m2"]), 0.05,
    disabled=not huerfanas, help="Más bajo = suma plantas chicas sueltas (y más ruido).")
st.markdown("**Limpieza de falsos positivos**")
c1, c2 = st.columns(2)
pc["green_min_peak_exg"] = c1.slider(
    "Verde mínimo de la mancha (ExG pico)", 0.05, 0.3, float(pc.get("green_min_peak_exg", 0.13)), 0.01,
    help="Descarta manchas cuyo píxel más verde no llega a este valor: rastrojo o suelo que apenas "
         "pasa el umbral. Una planta real tiene un núcleo de verde intenso. Más alto = más limpieza.")
pc["green_min_area_m2"] = c2.slider(
    "Área mínima de mancha (m²)", 0.02, 1.0, float(pc.get("green_min_area_m2", 0.1)), 0.02,
    help="Manchas más chicas se ignoran, tengan o no detección.")
keep_no_green = st.checkbox(
    "Conservar detecciones del modelo sin verde", value=False,
    help="Detecciones sin ninguna mancha verde válida. En barbecho suelen ser falsos positivos; "
         "activado, se poligonizan con GrabCut.")

final = raw[raw["confidence"] >= conf_thr].reset_index(drop=True)

# se recalcula solo cuando cambian las detecciones o los ajustes (segundos: la máscara de verde está cacheada)
key = hashlib.md5(final.geometry.bounds.to_numpy().tobytes() + final["confidence"].to_numpy().tobytes()
                  + repr((ortho_path, sorted(pc.items()), huerfanas, keep_no_green)).encode()).hexdigest()
if st.session_state.get("polys_key") != key:
    comps = cached_green(ortho_path, pc["exg_threshold"], pc["green_resolution_m"], pc["green_close_m"])
    with st.spinner("Poligonizando…"):
        st.session_state["polys"] = polygonize_green(final, ortho_path, dict(cfg, polygonize=pc), components=comps,
                                                     include_orphans=huerfanas, keep_no_green=keep_no_green)
    st.session_state["polys_key"] = key
polys = st.session_state["polys"]

cards([("Polígonos", len(polys)),
       ("Área de maleza", f"{polys['area_m2'].sum():.1f} m²" if len(polys) else "—"),
       ("Detectados por el modelo", int((polys["source"] != "exg").sum()) if len(polys) else 0,
        f"Polígonos que unen {len(final)} detecciones del modelo"),
       ("Verde sin detección", int((polys["source"] == "exg").sum()) if len(polys) else 0)], title="Métricas")
st.caption("Rojo: detectado por el modelo · Naranja: verde sin detección · "
           "las cajas del modelo se activan desde el selector de capas del mapa")
st_folium(build_map(ortho_path, final, polys, [], info["transform"], info["crs"], boxes_visible=False),
          height=650, use_container_width=True, returned_objects=[])

out_dir = os.path.join(cfg["paths"]["outputs_dir"], Path(ortho_path).stem)


def build_zip(final=final, polys=polys, out_dir=out_dir):
    """Se ejecuta recién al hacer click en Descargar (y escribe los archivos en outputs/)."""
    export(final.drop(columns=["px_x1", "px_y1", "px_x2", "px_y2", "truncated"], errors="ignore"), polys, out_dir)
    return zip_outputs(out_dir)


st.download_button("Descargar resultados (GPKG + GeoJSON + SHP)", build_zip, type="primary",
                   file_name=f"malezas_{Path(ortho_path).stem}.zip", mime="application/zip",
                   disabled=not len(polys))
st.caption(f"Polígonos + cajas del modelo. También quedan guardados en {out_dir}")
