import time
import psutil
import os
import cv2
from ultralytics import YOLO

def run_inference(model_path, video_path, conf_thresh, img_size):
    model = YOLO(model_path, task='detect')
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise FileNotFoundError(f"OpenCV no pudo abrir el video en: {video_path}")
        
    process = psutil.Process(os.getpid())
    results_data = []
    
    # Control de memoria: variables para retener solo un fotograma
    max_detections = 0
    best_plot = None 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        start_t = time.time()
        res = model(frame, conf=conf_thresh, imgsz=img_size, verbose=False)[0]
        
        latency = (time.time() - start_t) * 1000
        ram_usage = process.memory_info().rss / (1024 * 1024)
        num_detections = len(res.boxes)
        
        # Sobrescribe la imagen en RAM solo si encuentra más peces que el cuadro anterior
        if num_detections > max_detections:
            max_detections = num_detections
            best_plot = res.plot()
        
        results_data.append({
            'latency': latency,
            'ram': ram_usage
        })
        
    cap.release()
    
    # Devuelve una tupla separando los datos ligeros de la imagen pesada
    return results_data, best_plot