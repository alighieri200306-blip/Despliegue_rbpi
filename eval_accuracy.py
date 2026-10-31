"""Evalúa la precisión (mAP, precisión, recall, F1) de los 9 motores (3 versiones x 3 formatos).

Uso:
    python eval_accuracy.py --data /ruta/dataset/data.yaml --split test
    python eval_accuracy.py --data data.yaml --split test --device cpu --only Nano Small

Notas:
  * La precisión NO depende de la Raspberry: puedes correr este script en una PC usando los
    mismos archivos .pt / .onnx / .tflite que desplegarás en la Pi. Así evalúas el efecto real
    de la exportación sin esperar horas de CPU ARM. (Para .tflite la PC necesita tensorflow.)
  * Usa imágenes etiquetadas que NO se hayan usado para entrenar (split 'test' o 'val').
  * mAP se calcula con conf=0.001 (estándar). Precisión/recall/F1 son los del punto de mejor F1.
"""
import argparse
import csv
import os

import numpy as np

from config import CONFIG

FIELDS = ["size", "format", "map50", "map50_95", "precision", "recall", "f1", "file_mb", "status"]


def evaluate(model_path, data, split, imgsz, device, batch):
    from ultralytics import YOLO
    model = YOLO(model_path, task="detect")
    res = model.val(data=data, split=split, imgsz=imgsz, device=device, batch=batch,
                    conf=0.001, iou=0.6, plots=False, verbose=False)
    p, r = float(res.box.mp), float(res.box.mr)
    return {
        "map50": float(res.box.map50),
        "map50_95": float(res.box.map),
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if (p + r) > 0 else 0.0,
    }


def write_outputs(rows, out_dir, fps_by_key):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "accuracy_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else (round(r[k], 4) if isinstance(r[k], float) else r[k]))
                        for k in FIELDS})

    txt_path = os.path.join(out_dir, "accuracy_report.txt")
    with open(txt_path, "w") as f:
        f.write("Reporte de precisión por versión y formato\n" + "=" * 50 + "\n\n")
        f.write(f"{'Modelo':<8}{'Formato':<9}{'mAP50':>8}{'mAP50-95':>10}{'Prec':>8}{'Recall':>8}{'F1':>8}{'MB':>8}\n")
        for r in rows:
            if r["status"] != "ok":
                f.write(f"{r['size']:<8}{r['format']:<9}  {r['status']}\n")
                continue
            f.write(f"{r['size']:<8}{r['format']:<9}{r['map50']:>8.3f}{r['map50_95']:>10.3f}"
                    f"{r['precision']:>8.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}{r['file_mb']:>8.1f}\n")
    print(f"Guardado: {csv_path}\n          {txt_path}")

    # Gráfica de compromiso FPS vs mAP50 (requiere results_summary.csv del benchmark de velocidad)
    pts = [(r, fps_by_key[(r["size"], r["format"])]) for r in rows
           if r["status"] == "ok" and (r["size"], r["format"]) in fps_by_key]
    if not pts:
        print("Sin FPS del benchmark; se omite la gráfica FPS vs mAP.")
        return
    import matplotlib.pyplot as plt
    colors = {"Nano": "C0", "Small": "C1", "Medium": "C2"}
    markers = {"pytorch": "o", "onnx": "s", "tflite": "^"}
    fig, ax = plt.subplots(figsize=CONFIG["output_settings"]["plot_size"], layout="constrained")
    for r, fps in pts:
        ax.scatter(fps, r["map50"], c=colors.get(r["size"], "gray"),
                   marker=markers.get(r["format"], "o"), s=80)
        ax.annotate(f"{r['size']} {r['format']}", (fps, r["map50"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("FPS en Raspberry Pi 5 (solo detección)")
    ax.set_ylabel("mAP50")
    ax.set_title("Compromiso velocidad vs. precisión por versión y formato")
    ax.grid(alpha=0.3)
    base = os.path.join(CONFIG["output_settings"]["base_dir"], "graphs", "fps_vs_map")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    fig.savefig(base + ".png", dpi=CONFIG["output_settings"]["plot_dpi"])
    fig.savefig(base + ".svg", format="svg")
    plt.close(fig)
    print(f"Guardado: {base}.png")


def load_fps(report_dir):
    """FPS (modo detect) del benchmark, si ya existe results_summary.csv."""
    path = os.path.join(report_dir, "results_summary.csv")
    fps = {}
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["mode"] == "detect" and row["fps"]:
                    fps[(row["size"], row["format"])] = float(row["fps"])
    return fps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ruta al data.yaml del dataset etiquetado")
    ap.add_argument("--split", default="test", help="test o val (debe existir en el data.yaml)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=1, help="1 es lo más seguro para ONNX/TFLite exportados")
    ap.add_argument("--only", nargs="*", help="Limitar a ciertas versiones, p. ej. Nano Small")
    args = ap.parse_args()

    out_dir = os.path.join(CONFIG["output_settings"]["base_dir"], "reports")
    rows = []
    for size, fmts in CONFIG["models"].items():
        if args.only and size not in args.only:
            continue
        for fmt, path in fmts.items():
            print(f"\nEvaluando {size} / {fmt} ...")
            row = {"size": size, "format": fmt, "status": "ok",
                   "file_mb": os.path.getsize(path) / 1e6 if os.path.exists(path) else None}
            try:
                row.update(evaluate(path, args.data, args.split, CONFIG["img_size"], args.device, args.batch))
                print(f"   mAP50={row['map50']:.3f}  mAP50-95={row['map50_95']:.3f}  "
                      f"P={row['precision']:.3f}  R={row['recall']:.3f}")
            except Exception as e:
                row["status"] = f"error: {e}"
                print(f"   -> Omitido. {e}")
            rows.append(row)

    write_outputs(rows, out_dir, load_fps(out_dir))
