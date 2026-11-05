"""Entrenamiento YOLO (celdas 9, 20 y 21 del notebook) + seguimiento de corridas para la app."""
import csv
import os
import subprocess
import sys
import time

import yaml

from .common import resolve_device

RUNNING_IF_UPDATED_WITHIN_S = 15 * 60  # una época en CPU tarda ~3 min


def run_status(runs_dir):
    """Estado de cada corrida en runs/: época, mAP50 y si sigue corriendo (heurística por mtime)."""
    out = []
    for name in sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []:
        run = os.path.join(runs_dir, name)
        results, args = os.path.join(run, "results.csv"), os.path.join(run, "args.yaml")
        if not os.path.exists(args):
            continue
        with open(args) as f:
            total = yaml.safe_load(f).get("epochs")
        rows = []
        if os.path.exists(results):
            with open(results) as f:
                rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
        stamps = [os.path.getmtime(p) for p in (results, args, os.path.join(run, "weights", "last.pt"))
                  if os.path.exists(p)]
        last = rows[-1] if rows else {}
        best = max((float(r.get("metrics/mAP50(B)", 0)) for r in rows), default=None)
        out.append({
            "name": name, "epoch": int(float(last["epoch"])) if last else 0, "epochs": total,
            "map50_best": best,
            # sin cambios en los últimos 15 min = terminó (o se cortó)
            "running": bool(stamps) and time.time() - max(stamps) < RUNNING_IF_UPDATED_WITHIN_S,
        })
    return out


def launch_training(config_path, name, weights=None, epochs=None):
    """Lanza `maleza train --rebuild` como proceso independiente (sigue aunque se cierre la app)."""
    root = os.path.dirname(os.path.abspath(config_path))
    log_path = os.path.join(root, "runs", f"{name}_train.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = [sys.executable, "-m", "maleza", "--config", config_path, "train", "--name", name, "--rebuild"]
    if weights:
        cmd += ["--weights", weights]
    if epochs:
        cmd += ["--epochs", str(epochs)]
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS) if os.name == "nt" else 0
    with open(log_path, "w") as log:
        subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
                         start_new_session=os.name != "nt")
    return log_path


def train(cfg, name, weights=None, data_yaml=None, **overrides):
    """weights=None arranca de los pesos COCO de cfg.train.model; si no, continúa desde ese .pt.
    data_yaml: default <paths.dataset_dir>/data.yaml."""
    from ultralytics import YOLO

    data_yaml = data_yaml or os.path.join(cfg["paths"]["dataset_dir"], "data.yaml")
    if not os.path.exists(data_yaml):
        raise SystemExit(f"No existe {data_yaml}: corré primero `maleza dataset`.")

    model = YOLO(weights or cfg["train"]["model"])
    params = {**cfg["train"]["params"], **overrides}
    return model.train(
        data=data_yaml,
        project=cfg["paths"]["runs_dir"],
        name=name,
        device=resolve_device(cfg["train"]["device"]),
        **params,
    )
