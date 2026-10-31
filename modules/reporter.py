import csv
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from modules import sysmon

MODE_LABELS = {"detect": "Solo detección", "track": "Detección + ByteTrack"}

SUMMARY_FIELDS = [
    "mode", "size", "format", "frames",
    "mean_latency_ms", "std_latency_ms", "p95_latency_ms", "fps", "ram_mb",
    "temp_mean_c", "temp_max_c", "freq_mean_mhz", "freq_min_mhz",
    "cpu_mean_pct", "throttled_now_mask", "mean_detections",
    "unique_ids", "mean_track_len_frames", "mean_active_tracks", "std_active_tracks",
]
FRAME_FIELDS = [
    "mode", "size", "format", "frame", "t_s", "latency_ms", "ram_mb",
    "n_det", "n_tracks", "temp_c", "freq_mhz", "cpu_pct", "throttled_now",
]


# ---------------------------------------------------------------- resumen
def _vals(metrics, key):
    return [m[key] for m in metrics if m.get(key) is not None]


def summarize(metrics, mode):
    """Resume la lista de metricas por frame de una combinacion modelo/formato."""
    if not metrics:
        return None

    lats = np.array([m['latency'] for m in metrics])
    mean_lat = float(lats.mean())
    s = {
        "frames": len(lats),
        "mean_lat": mean_lat,
        "std_lat": float(lats.std()),
        "p95_lat": float(np.percentile(lats, 95)),
        "fps": 1000 / mean_lat if mean_lat > 0 else 0.0,
        "ram": float(np.mean([m['ram'] for m in metrics])),
        "mean_det": float(np.mean([m['n_det'] for m in metrics])),
    }

    temps = _vals(metrics, 'temp_c')
    freqs = _vals(metrics, 'freq_mhz')
    cpus = _vals(metrics, 'cpu_pct')
    thr = _vals(metrics, 'throttled_now')
    s["temp_mean"] = float(np.mean(temps)) if temps else None
    s["temp_max"] = float(np.max(temps)) if temps else None
    s["freq_mean"] = float(np.mean(freqs)) if freqs else None
    s["freq_min"] = float(np.min(freqs)) if freqs else None
    s["cpu_mean"] = float(np.mean(cpus)) if cpus else None
    mask = 0
    for v in thr:
        mask |= v
    s["throttled"] = mask if thr else None

    if mode == "track":
        counts = Counter()
        active = []
        for m in metrics:
            counts.update(m['ids'])
            active.append(len(m['ids']))
        s["unique_ids"] = len(counts)
        s["mean_track_len"] = float(np.mean(list(counts.values()))) if counts else 0.0
        s["mean_active"] = float(np.mean(active))
        s["std_active"] = float(np.std(active))
    return s


def _f(value, spec, unit=""):
    return "N/D" if value is None else f"{value:{spec}}{unit}"


# ---------------------------------------------------------------- texto
def _write_mode_report(path, mode, summaries, sizes, formats):
    with open(path, 'w') as f:
        f.write(f"Reporte de Evaluación por Versión y Formato - {MODE_LABELS.get(mode, mode)}\n")
        f.write("=" * 50 + "\n\n")
        for size in sizes:
            f.write(f"--- VERSIÓN: {size.upper()} ---\n")
            for fmt in formats:
                s = summaries[size].get(fmt)
                if not s:
                    f.write(f"  Formato {fmt}: Sin datos.\n")
                    continue
                f.write(f"  Formato: {fmt}\n")
                f.write(f"    Latencia Media: {s['mean_lat']:.2f} ms (desv. {s['std_lat']:.2f}, p95 {s['p95_lat']:.2f})\n")
                f.write(f"    FPS Promedio:   {s['fps']:.2f}\n")
                f.write(f"    RAM Promedio:   {s['ram']:.2f} MB\n")
                f.write(f"    CPU Promedio:   {_f(s['cpu_mean'], '.1f', ' %')}\n")
                f.write(f"    Temp (media/máx): {_f(s['temp_mean'], '.1f')} / {_f(s['temp_max'], '.1f', ' °C')}\n")
                f.write(f"    Frecuencia CPU (media/mín): {_f(s['freq_mean'], '.0f')} / {_f(s['freq_min'], '.0f', ' MHz')}\n")
                f.write(f"    Throttling durante la corrida: {sysmon.describe_throttle(s['throttled'])}\n")
                f.write(f"    Detecciones por frame: {s['mean_det']:.2f}\n")
                if mode == "track":
                    f.write(f"    IDs únicos asignados: {s['unique_ids']}\n")
                    f.write(f"    Duración media de trayectoria: {s['mean_track_len']:.1f} frames\n")
                    f.write(f"    Tracks activos por frame (media ± desv.): {s['mean_active']:.2f} ± {s['std_active']:.2f}\n")
            f.write("\n")


def _write_comparison_report(path, summaries_by_mode, sizes, formats):
    det = summaries_by_mode.get("detect")
    trk = summaries_by_mode.get("track")
    if not det or not trk:
        return
    with open(path, 'w') as f:
        f.write("Comparativa: solo detección vs. detección + ByteTrack\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Modelo':<8}{'Formato':<9}{'Lat det':>10}{'Lat track':>11}"
                f"{'Overhead':>10}{'%':>7}{'FPS det':>9}{'FPS trk':>9}{'Tmax det':>10}{'Tmax trk':>10}\n")
        for size in sizes:
            for fmt in formats:
                d, t = det[size].get(fmt), trk[size].get(fmt)
                if not d or not t:
                    f.write(f"{size:<8}{fmt:<9}  Sin datos en alguno de los modos.\n")
                    continue
                over = t['mean_lat'] - d['mean_lat']
                pct = 100 * over / d['mean_lat'] if d['mean_lat'] > 0 else 0
                f.write(f"{size:<8}{fmt:<9}{d['mean_lat']:>10.2f}{t['mean_lat']:>11.2f}"
                        f"{over:>10.2f}{pct:>7.1f}{d['fps']:>9.2f}{t['fps']:>9.2f}"
                        f"{_f(d['temp_max'], '.1f'):>10}{_f(t['temp_max'], '.1f'):>10}\n")
        f.write("\nLatencias en ms; Overhead = lat(track) - lat(detect); temperaturas en °C.\n")
        f.write("Nota: si track_conf_threshold != conf_threshold, el overhead incluye también el\n"
                "costo de procesar más cajas de baja confianza, no solo el del tracker.\n")


# ---------------------------------------------------------------- CSV
def _round(v, n):
    return "" if v is None else round(v, n)


def _write_csvs(report_dir, all_metrics, summaries_by_mode, sizes, formats):
    with open(os.path.join(report_dir, "results_summary.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for mode, summaries in summaries_by_mode.items():
            for size in sizes:
                for fmt in formats:
                    s = summaries[size].get(fmt)
                    if not s:
                        continue
                    w.writerow({
                        "mode": mode, "size": size, "format": fmt, "frames": s["frames"],
                        "mean_latency_ms": round(s["mean_lat"], 2),
                        "std_latency_ms": round(s["std_lat"], 2),
                        "p95_latency_ms": round(s["p95_lat"], 2),
                        "fps": round(s["fps"], 3), "ram_mb": round(s["ram"], 2),
                        "temp_mean_c": _round(s["temp_mean"], 1), "temp_max_c": _round(s["temp_max"], 1),
                        "freq_mean_mhz": _round(s["freq_mean"], 0), "freq_min_mhz": _round(s["freq_min"], 0),
                        "cpu_mean_pct": _round(s["cpu_mean"], 1),
                        "throttled_now_mask": "" if s["throttled"] is None else hex(s["throttled"]),
                        "mean_detections": round(s["mean_det"], 2),
                        "unique_ids": s.get("unique_ids", ""),
                        "mean_track_len_frames": _round(s.get("mean_track_len"), 1),
                        "mean_active_tracks": _round(s.get("mean_active"), 2),
                        "std_active_tracks": _round(s.get("std_active"), 2),
                    })

    with open(os.path.join(report_dir, "results_per_frame.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FRAME_FIELDS)
        w.writeheader()
        for mode, by_size in all_metrics.items():
            for size in sizes:
                for fmt in formats:
                    for i, m in enumerate(by_size.get(size, {}).get(fmt) or []):
                        w.writerow({
                            "mode": mode, "size": size, "format": fmt, "frame": i,
                            "t_s": round(m['t_s'], 3), "latency_ms": round(m['latency'], 2),
                            "ram_mb": round(m['ram'], 2), "n_det": m['n_det'], "n_tracks": len(m['ids']),
                            "temp_c": "" if m['temp_c'] is None else m['temp_c'],
                            "freq_mhz": "" if m['freq_mhz'] is None else m['freq_mhz'],
                            "cpu_pct": "" if m['cpu_pct'] is None else m['cpu_pct'],
                            "throttled_now": "" if m['throttled_now'] is None else m['throttled_now'],
                        })


# ---------------------------------------------------------------- graficas
def _save(fig, base_path, dpi):
    fig.savefig(f"{base_path}.png", dpi=dpi)
    fig.savefig(f"{base_path}.svg", format='svg')
    plt.close(fig)


def _fps_by_format(summaries, sizes, formats):
    return {fmt: [(summaries[size].get(fmt) or {}).get('fps', 0) for size in sizes] for fmt in formats}


def _plot_fps_mode(mode, summaries, sizes, formats, out_settings, graph_dir):
    fps_data = _fps_by_format(summaries, sizes, formats)
    x = np.arange(len(sizes))
    width = 0.25
    fig, ax = plt.subplots(figsize=out_settings['plot_size'], layout='constrained')
    for i, (fmt, vals) in enumerate(fps_data.items()):
        rects = ax.bar(x + width * i, vals, width, label=fmt)
        ax.bar_label(rects, padding=3, fmt='%.1f')
    ax.set_ylabel('Cuadros por Segundo (FPS)')
    ax.set_title(f'Comparativa de FPS por Tamaño de Modelo y Formato ({MODE_LABELS.get(mode, mode)})')
    ax.set_xticks(x + width, sizes)
    ax.legend(loc='upper left', ncols=3)
    max_fps = max([max(v) for v in fps_data.values()]) if fps_data else 0
    ax.set_ylim(0, (max_fps * 1.2) or 1)
    _save(fig, os.path.join(graph_dir, f"{out_settings['graph_name']}_{mode}"), out_settings['plot_dpi'])


def _plot_detect_vs_track(summaries_by_mode, sizes, formats, out_settings, graph_dir):
    det_fps = _fps_by_format(summaries_by_mode["detect"], sizes, formats)
    trk_fps = _fps_by_format(summaries_by_mode["track"], sizes, formats)
    x = np.arange(len(sizes))
    width = 0.38
    fig, axes = plt.subplots(1, len(formats), figsize=(5 * len(formats), 5),
                             sharey=True, layout='constrained')
    axes = np.atleast_1d(axes)
    for ax, fmt in zip(axes, formats):
        r1 = ax.bar(x - width / 2, det_fps[fmt], width, label=MODE_LABELS["detect"])
        r2 = ax.bar(x + width / 2, trk_fps[fmt], width, label=MODE_LABELS["track"])
        ax.bar_label(r1, padding=2, fmt='%.2f', fontsize=8)
        ax.bar_label(r2, padding=2, fmt='%.2f', fontsize=8)
        ax.set_title(fmt)
        ax.set_xticks(x, sizes)
    axes[0].set_ylabel('Cuadros por Segundo (FPS)')
    axes[0].legend(loc='upper right', fontsize=8)
    fig.suptitle('FPS: solo detección vs. detección + ByteTrack')
    _save(fig, os.path.join(graph_dir, "fps_detect_vs_track"), out_settings['plot_dpi'])


def _plot_temperature(mode, all_metrics, sizes, formats, out_settings, graph_dir):
    colors = {size: f"C{i}" for i, size in enumerate(sizes)}
    styles = dict(zip(formats, ['-', '--', ':', '-.']))
    fig, ax = plt.subplots(figsize=out_settings['plot_size'], layout='constrained')
    plotted = False
    for size in sizes:
        for fmt in formats:
            rows = (all_metrics[mode].get(size, {}).get(fmt)) or []
            pts = [(m['t_s'], m['temp_c']) for m in rows if m.get('temp_c') is not None]
            if not pts:
                continue
            t, temp = zip(*pts)
            ax.plot(t, temp, styles.get(fmt, '-'), color=colors[size], label=f"{size} {fmt}")
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(80, color='gray', linestyle='--', linewidth=0.8)
    ax.text(0.01, 80, ' ~80 °C: inicio de throttling', transform=ax.get_yaxis_transform(),
            va='bottom', fontsize=8, color='gray')
    ax.set_xlabel('Tiempo de la corrida (s)')
    ax.set_ylabel('Temperatura del SoC (°C)')
    ax.set_title(f'Temperatura durante la inferencia ({MODE_LABELS.get(mode, mode)})')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
    _save(fig, os.path.join(graph_dir, f"temperature_{mode}"), out_settings['plot_dpi'])


# ---------------------------------------------------------------- punto de entrada
def generate_reports_and_graphs(all_metrics, out_settings):
    """all_metrics[modo][tamaño][formato] -> lista de metricas por frame."""
    base_dir = out_settings['base_dir']
    report_dir = os.path.join(base_dir, "reports")
    graph_dir = os.path.join(base_dir, "graphs")
    report_base, report_ext = os.path.splitext(out_settings['report_name'])

    modes = list(all_metrics.keys())
    sizes = list(all_metrics[modes[0]].keys())
    formats = sorted({fmt for mode in modes for size in sizes for fmt in all_metrics[mode].get(size, {})},
                     key=lambda x: ['pytorch', 'onnx', 'tflite'].index(x) if x in ['pytorch', 'onnx', 'tflite'] else 99)

    summaries_by_mode = {}
    for mode in modes:
        summaries = {size: {fmt: summarize(all_metrics[mode].get(size, {}).get(fmt), mode)
                            for fmt in formats} for size in sizes}
        summaries_by_mode[mode] = summaries
        _write_mode_report(os.path.join(report_dir, f"{report_base}_{mode}{report_ext}"),
                           mode, summaries, sizes, formats)
        _plot_fps_mode(mode, summaries, sizes, formats, out_settings, graph_dir)
        _plot_temperature(mode, all_metrics, sizes, formats, out_settings, graph_dir)

    _write_comparison_report(os.path.join(report_dir, f"{report_base}_comparison{report_ext}"),
                             summaries_by_mode, sizes, formats)
    if "detect" in summaries_by_mode and "track" in summaries_by_mode:
        _plot_detect_vs_track(summaries_by_mode, sizes, formats, out_settings, graph_dir)
    _write_csvs(report_dir, all_metrics, summaries_by_mode, sizes, formats)
