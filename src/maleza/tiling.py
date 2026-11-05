"""Info y tiling de ortomosaicos (celdas 4, 5, 16 y 17 del notebook)."""
import json
import os

import rasterio
from PIL import Image

from .common import iter_windows, metadata_filename, read_tile, tile_name


def ortho_info(ortho_path):
    with rasterio.open(ortho_path) as src:
        print(f"--- {os.path.basename(ortho_path)} ---")
        print(f"  Tamaño: {src.width}x{src.height} | Bandas: {src.count} | Dtype: {src.dtypes[0]}")
        print(f"  CRS: {src.crs} | Resolución (m/px): {src.res}")


def tile_orthomosaic(ortho_path, prefix, output_dir, tile_size=1024, overlap=200,
                     nodata_threshold=0.3):
    os.makedirs(output_dir, exist_ok=True)
    tile_metadata = []
    with rasterio.open(ortho_path) as src:
        tile_id = 0
        for window in iter_windows(src.width, src.height, tile_size, overlap):
            tile, black_frac = read_tile(src, window, tile_size)
            # descartar tiles mayormente vacíos (bordes irregulares del ortomosaico)
            if black_frac > nodata_threshold:
                continue

            filename = f"{tile_name(prefix, tile_id)}.jpg"
            Image.fromarray(tile).save(os.path.join(output_dir, filename), quality=95)
            tile_metadata.append({
                "tile_id": tile_id, "filename": filename,
                "window_left": int(window.col_off), "window_top": int(window.row_off),
                "window_width": int(window.width), "window_height": int(window.height),
                "transform": list(src.window_transform(window))[:6],
                "crs": str(src.crs),
            })
            tile_id += 1

    with open(os.path.join(output_dir, metadata_filename(prefix)), "w") as f:
        json.dump(tile_metadata, f, indent=2)
    print(f"{prefix or 'original'}: {tile_id} tiles generados")
