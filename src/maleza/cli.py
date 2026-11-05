"""CLI: maleza <comando> [--config config.yaml]"""
import argparse
import os
from pathlib import Path

from .common import load_config


def _selected_orthos(cfg, names):
    orthos = cfg["orthos"]
    if names:
        orthos = [o for o in orthos if (o["prefix"] or "o1") in names or Path(o["path"]).stem in names]
    return orthos


def cmd_info(cfg, args):
    from .tiling import ortho_info
    for o in _selected_orthos(cfg, args.orthos):
        ortho_info(o["path"]) if os.path.exists(o["path"]) else print(f"No existe: {o['path']}")


def cmd_tile(cfg, args):
    from .tiling import tile_orthomosaic
    t = cfg["tiling"]
    for o in _selected_orthos(cfg, args.orthos):
        tile_orthomosaic(o["path"], o["prefix"], cfg["paths"]["tiles_dir"],
                         t["tile_size"], t["overlap"], t["nodata_threshold"])


def cmd_dataset(cfg, args):
    from .dataset import build_dataset
    build_dataset(cfg)


def cmd_train(cfg, args):
    from .train import train
    overrides = {"epochs": args.epochs} if args.epochs else {}
    data_yaml = None
    if args.rebuild:
        # dataset propio de esta corrida: no pisa data/dataset de otro entrenamiento en curso
        from .dataset import build_dataset
        data_yaml = build_dataset(cfg, os.path.join(cfg["root"], "data", "datasets", args.name))
    train(cfg, args.name, weights=args.weights, data_yaml=data_yaml, **overrides)


def cmd_predict(cfg, args):
    from .polygonize import export, polygonize
    from .predict import detect_ortho, load_model

    weights = args.weights or cfg["predict"]["weights"]
    model = load_model(weights)
    print(f"Modelo: {weights}")
    paths = args.tif or [o["path"] for o in _selected_orthos(cfg, args.orthos)]
    for ortho_path in paths:
        print(f"\n=== {os.path.basename(ortho_path)} ===")
        gdf_boxes, n_raw = detect_ortho(model, ortho_path, cfg)
        print(f"Detecciones crudas: {n_raw} | Después de deduplicar: {len(gdf_boxes)}")
        gdf_polys = polygonize(gdf_boxes, ortho_path, cfg) if len(gdf_boxes) else gdf_boxes.assign(area_m2=[])
        export(gdf_boxes, gdf_polys, os.path.join(cfg["paths"]["outputs_dir"], Path(ortho_path).stem))


def cmd_inspect(cfg, args):
    from .predict import inspect_tile, load_model
    inspect_tile(load_model(args.weights or cfg["predict"]["weights"]), args.tile, args.conf)


def cmd_calibrar(cfg, args):
    import pandas as pd
    from .calibrate import calibrate
    o = _selected_orthos(cfg, args.orthos or ["o1"])[0]
    df = calibrate(cfg, o["path"], o["prefix"] or "o1", args.weights or cfg["predict"]["weights"])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\nMejores 15 por F1 (equilibrio entre recall y precisión):")
    print(df.head(15).round(3).to_string(index=False))


def cmd_app(cfg, args):
    import sys
    if sys.stdout is None:  # pythonw (sin consola): los mensajes y errores van a outputs/app.log
        os.makedirs(cfg["paths"]["outputs_dir"], exist_ok=True)
        sys.stdout = sys.stderr = open(os.path.join(cfg["paths"]["outputs_dir"], "app.log"), "w",
                                       encoding="utf-8", buffering=1)
    from .app import main as app_main
    sys.exit(app_main(cfg, Path(args.config).resolve()))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="maleza", description="Detección de malezas en ortomosaicos")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="resolución / CRS / tamaño de los ortomosaicos")
    p.add_argument("orthos", nargs="*", help="prefijos (o1, o2...) o nombres; vacío = todos")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("tile", help="generar tiles JPG + metadata para etiquetar")
    p.add_argument("orthos", nargs="*")
    p.set_defaults(func=cmd_tile)

    p = sub.add_parser("dataset", help="split espacial train/val + sintéticos + data.yaml")
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("train", help="entrenar YOLO")
    p.add_argument("--name", required=True, help="nombre de la corrida (runs/<name>)")
    p.add_argument("--weights", help="continuar desde este .pt (default: pesos COCO)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--rebuild", action="store_true",
                   help="armar un dataset nuevo en data/datasets/<name> con todo el pool etiquetado")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="detectar + poligonizar + exportar")
    p.add_argument("orthos", nargs="*", help="prefijos/nombres del config; vacío = todos")
    p.add_argument("--tif", nargs="+", help="rutas a .tif fuera del config")
    p.add_argument("--weights")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("inspect", help="confianzas crudas en un tile (diagnóstico)")
    p.add_argument("tile")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--weights")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("calibrar", help="medir los ajustes de la app contra las etiquetas a mano")
    p.add_argument("orthos", nargs="*", help="prefijo del orto etiquetado (default: o1)")
    p.add_argument("--weights")
    p.set_defaults(func=cmd_calibrar)

    p = sub.add_parser("app", help="abrir la app de escritorio (cargar ortomosaico y detectar)")
    p.set_defaults(func=cmd_app)

    args = parser.parse_args(argv)
    args.func(load_config(args.config), args)


if __name__ == "__main__":
    main()
