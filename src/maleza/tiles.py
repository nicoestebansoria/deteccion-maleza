"""Servidor local de teselas XYZ del ortomosaico, para verlo en el mapa con su resolución original.

La vista previa (mapa.ortho_preview) es una sola imagen reducida y alcanza para zooms lejanos; de cerca,
Leaflet pide teselas de 256 px a http://127.0.0.1:<puerto>/<clave>/{z}/{x}/{y}.webp y cada una se lee del
.tif reproyectada a EPSG:3857 (WarpedVRT), solo con el detalle que necesita ese zoom.
"""
import hashlib
import io
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

TILE = 256
ORIGIN = math.pi * 6378137  # medio ancho del mundo en Web Mercator (m)


def tile_bounds(z, x, y):
    size = 2 * ORIGIN / 2 ** z
    minx, maxy = -ORIGIN + x * size, ORIGIN - y * size
    return minx, maxy - size, minx + size, maxy


class TileServer:
    def __init__(self):
        self.orthos = {}  # clave -> {"path", "bounds" (3857), "res" (m/px), "stretch"}
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    key, z, x, y = self.path.strip("/").removesuffix(".webp").split("/")
                    body = server.render(key, int(z), int(x), int(y))
                except Exception:
                    body = None
                if body is None:  # fuera del orto o sin datos: Leaflet no dibuja nada
                    self.send_response(404)
                    self.end_headers()
                    return
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/webp")
                    self.send_header("Cache-Control", "max-age=3600")
                    self.end_headers()
                    self.wfile.write(body)
                except ConnectionError:  # Leaflet cancela las teselas que dejan de verse al mover el mapa
                    pass

            def log_message(self, *args):
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):  # conexiones cortadas: no ensuciar el log
                pass

        self.httpd = Server(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def register(self, path):
        """Devuelve (url de teselas para Leaflet, zoom nativo máximo)."""
        key = hashlib.md5(path.encode()).hexdigest()[:12]
        if key not in self.orthos:
            with rasterio.open(path) as src:
                bounds = transform_bounds(src.crs, "EPSG:3857", *src.bounds)
                # resolución en metros sobre el terreno (en ortos geográficos se aproxima desde grados)
                res = src.res[0] * (111_320 if src.crs.is_geographic else 1)
                stretch = None
                if src.dtypes[0] != "uint8":  # 16 bits: estirar al rango 2-98% (igual que la vista previa)
                    sample = src.read([1, 2, 3], out_shape=(3, 512, 512))
                    lo, hi = np.percentile(sample[sample > 0], [2, 98]) if (sample > 0).any() else (0, 1)
                    stretch = (float(lo), float(hi))
            self.orthos[key] = {"path": path, "bounds": bounds, "res": res, "stretch": stretch}
        o = self.orthos[key]
        lat = math.degrees(math.atan(math.sinh(((o["bounds"][1] + o["bounds"][3]) / 2) / ORIGIN)))
        # primer zoom cuyo píxel en el terreno es igual o más chico que el del orto
        native_zoom = math.ceil(math.log2(2 * ORIGIN * math.cos(math.radians(lat)) / (TILE * o["res"])))
        return f"http://127.0.0.1:{self.httpd.server_port}/{key}/{{z}}/{{x}}/{{y}}.webp", native_zoom

    def render(self, key, z, x, y):
        o = self.orthos.get(key)
        if o is None:
            return None
        minx, miny, maxx, maxy = tile_bounds(z, x, y)
        b = o["bounds"]
        if maxx <= b[0] or minx >= b[2] or maxy <= b[1] or miny >= b[3]:
            return None
        px = (maxx - minx) / TILE
        # alejado: promedio de los píxeles que caen en cada píxel de la tesela; de cerca, bilineal
        lat_scale = math.cos(math.atan(math.sinh(((miny + maxy) / 2) / ORIGIN)))
        resampling = Resampling.average if px * lat_scale > o["res"] * 1.5 else Resampling.bilinear
        with rasterio.open(o["path"]) as src, WarpedVRT(
                src, crs="EPSG:3857", transform=from_origin(minx, maxy, px, px), width=TILE, height=TILE,
                resampling=resampling) as vrt:
            alpha = vrt.dataset_mask()
            if not alpha.any():
                return None
            rgb = np.moveaxis(vrt.read([1, 2, 3]), 0, -1)
        if o["stretch"]:
            lo, hi = o["stretch"]
            rgb = (np.clip((rgb - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(np.dstack([rgb.astype(np.uint8), alpha])).save(buf, format="WEBP", quality=88)
        return buf.getvalue()

    def close(self):
        self.httpd.shutdown()
