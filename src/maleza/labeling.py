"""Etiquetado / revisión con BBoxWidget en Jupyter local (celdas 6, 22 y 23 del notebook).

Uso en un notebook:
    from maleza.labeling import etiquetar
    etiquetar(tiles, labels_dir)                  # todos los tiles, arranca por el primero sin label
    etiquetar(tiles, labels_dir, solo_pendientes=False, progress_path="revision.json")  # revisión
"""
import base64
import json
import os

from .common import load_yolo_labels, write_yolo_labels


def _image_to_base64(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")


def etiquetar(tiles, labels_dir, solo_pendientes=True, progress_path=None, tile_w=1024, tile_h=1024):
    """Muestra el widget. Carga las cajas existentes y autoguarda cada edición."""
    import ipywidgets as widgets
    from IPython.display import display
    from jupyter_bbox_widget import BBoxWidget

    os.makedirs(labels_dir, exist_ok=True)
    tiles = sorted(tiles)

    def label_path(i):
        name = os.path.splitext(os.path.basename(tiles[i]))[0]
        return os.path.join(labels_dir, f"{name}.txt")

    if progress_path and os.path.exists(progress_path):
        with open(progress_path) as f:
            start = json.load(f).get("last_index", 0)
    elif solo_pendientes:
        start = next((i for i in range(len(tiles)) if not os.path.exists(label_path(i))), len(tiles))
    else:
        start = 0

    state = {"i": start, "loading": False}
    widget = BBoxWidget(image="", classes=["maleza"])
    progress = widgets.Label()
    status = widgets.Label()
    save_btn = widgets.Button(description="Guardar y siguiente", button_style="success")
    skip_btn = widgets.Button(description="Sin maleza (vacío)", button_style="warning")

    def save(bboxes):
        boxes = [((b["x"] + b["width"] / 2) / tile_w, (b["y"] + b["height"] / 2) / tile_h,
                  b["width"] / tile_w, b["height"] / tile_h) for b in bboxes]
        write_yolo_labels(label_path(state["i"]), boxes)
        status.value = f"✓ Guardado ({len(boxes)} cajas)"

    def load():
        if state["i"] >= len(tiles):
            progress.value = f"¡Completo! {len(tiles)} tiles en {labels_dir}"
            widget.image, widget.bboxes = "", []
            return
        state["loading"] = True  # evita que el autoguardado dispare al cambiar de tile
        widget.image = _image_to_base64(tiles[state["i"]])
        widget.bboxes = [
            {"x": (cx - w / 2) * tile_w, "y": (cy - h / 2) * tile_h,
             "width": w * tile_w, "height": h * tile_h, "label": "maleza"}
            for cx, cy, w, h in load_yolo_labels(label_path(state["i"]))
        ]
        state["loading"] = False
        progress.value = (f"Tile {state['i'] + 1} / {len(tiles)} — "
                          f"{os.path.basename(tiles[state['i']])} ({len(widget.bboxes)} cajas)")
        status.value = ""
        if progress_path:
            with open(progress_path, "w") as f:
                json.dump({"last_index": state["i"]}, f)

    def on_change(change):
        if not state["loading"] and state["i"] < len(tiles):
            save(widget.bboxes)

    def advance(bboxes):
        if state["i"] < len(tiles):
            save(bboxes)
            state["i"] += 1
            load()

    widget.observe(on_change, names="bboxes")
    save_btn.on_click(lambda _: advance(widget.bboxes))
    skip_btn.on_click(lambda _: advance([]))
    display(progress, widget, widgets.HBox([save_btn, skip_btn]), status)
    load()
    return widget
