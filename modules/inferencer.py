import time
import psutil
import os
import cv2
from ultralytics import YOLO

from modules import sysmon


def run_inference(model_path, video_path, conf_thresh, img_size,
                  mode="detect", tracker_cfg=None,
                  warmup_frames=0, sample_every=10):
    """Corre un modelo sobre el video.

    mode="detect": model(frame)                       -> solo deteccion
    mode="track":  model.track(frame, persist=True)   -> deteccion + ByteTrack

    La latencia medida con perf_counter incluye preproceso, inferencia,
    postproceso y (en modo track) la actualizacion del tracker.
    Devuelve (lista_de_metricas_por_frame, mejor_frame).
    """
    model = YOLO(model_path, task='detect')
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"OpenCV no pudo abrir el video en: {video_path}")
    if mode == "track" and not tracker_cfg:
        raise ValueError("El modo 'track' requiere tracker_cfg (ruta al YAML de ByteTrack).")

    process = psutil.Process(os.getpid())
    psutil.cpu_percent(interval=None)  # inicializa el contador de CPU
    results_data = []

    # Control de memoria: variables para retener solo un fotograma
    max_detections = 0
    best_plot = None

    frame_idx = 0
    n_measured = 0
    t_run0 = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_t = time.perf_counter()
        if mode == "track":
            res = model.track(frame, persist=True, tracker=tracker_cfg,
                              conf=conf_thresh, imgsz=img_size, verbose=False)[0]
        else:
            res = model(frame, conf=conf_thresh, imgsz=img_size, verbose=False)[0]
        latency = (time.perf_counter() - start_t) * 1000

        frame_idx += 1
        if frame_idx <= warmup_frames:
            continue  # calentamiento: se ejecuta pero no se registra

        ram_usage = process.memory_info().rss / (1024 * 1024)
        num_detections = len(res.boxes)
        track_ids = []
        if mode == "track" and res.boxes.id is not None:
            track_ids = res.boxes.id.int().tolist()

        # Sobrescribe la imagen en RAM solo si encuentra más peces que el cuadro anterior
        if num_detections > max_detections:
            max_detections = num_detections
            best_plot = res.plot()

        row = {
            'latency': latency,
            'ram': ram_usage,
            't_s': time.perf_counter() - t_run0,
            'n_det': num_detections,
            'ids': track_ids,
            'temp_c': None,
            'freq_mhz': None,
            'throttled_now': None,
            'cpu_pct': None,
        }

        # Lectura periodica del sistema (leer cada frame agregaria overhead)
        if n_measured % sample_every == 0:
            row.update(sysmon.snapshot())
            row['cpu_pct'] = psutil.cpu_percent(interval=None)

        results_data.append(row)
        n_measured += 1

    cap.release()

    # Devuelve una tupla separando los datos ligeros de la imagen pesada
    return results_data, best_plot
