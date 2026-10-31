import os

# Obtiene la ruta base absoluta del directorio actual para evitar errores de rutas relativas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # 1. Rutas de los Modelos (Organizados por tamaño y formato)
    "models": {
        "Nano": {
            "pytorch": os.path.join(BASE_DIR, "models", "Nano", "fish_model.pt"),
            "onnx": os.path.join(BASE_DIR, "models", "Nano", "fish_model.onnx"),
            "tflite": os.path.join(BASE_DIR, "models", "Nano", "fish_model.tflite")
        },
        "Small": {
            "pytorch": os.path.join(BASE_DIR, "models", "Small", "fish_model.pt"),
            "onnx": os.path.join(BASE_DIR, "models", "Small", "fish_model.onnx"),
            "tflite": os.path.join(BASE_DIR, "models", "Small", "fish_model.tflite")
        },
        "Medium": {
            "pytorch": os.path.join(BASE_DIR, "models", "Medium", "fish_model.pt"),
            "onnx": os.path.join(BASE_DIR, "models", "Medium", "fish_model.onnx"),
            "tflite": os.path.join(BASE_DIR, "models", "Medium", "fish_model.tflite")
        }
    },

    # 2. Ruta del video de entrada
    "video_path": os.path.join(BASE_DIR, "videos", "Prueba_3min.mp4"),

    # 3. Parámetros de Inferencia
    # Umbral de confianza (ajustado a 0.25 para ayudar a Nano y Small a detectar)
    "conf_threshold": 0.25,
    # Tamaño de la imagen para la red neuronal (evita el colapso del motor ONNX)
    "img_size": 640,

    # 4. Modos de evaluación: primero solo detección, luego detección + ByteTrack
    "modes": ["detect", "track"],

    # 5. Configuración de ByteTrack
    "tracker_cfg": os.path.join(BASE_DIR, "trackers", "bytetrack_fish.yaml"),
    # ByteTrack usa en su 2a asociacion detecciones de baja confianza. Si el umbral de
    # inferencia es 0.25, esas detecciones nunca llegan al tracker; por eso en modo "track"
    # se infiere con un umbral menor (= track_low_thresh del YAML). Pon 0.25 aqui si quieres
    # que detect y track usen exactamente el mismo umbral al medir el overhead.
    "track_conf_threshold": 0.1,

    # 6. Medicion del sistema
    # Frames iniciales que se ejecutan pero no se promedian (calentamiento del motor).
    "warmup_frames": 10,
    # Cada cuantos frames se lee temperatura / frecuencia / throttling / CPU.
    "sample_every": 10,
    "thermal": {
        # Antes de cada corrida se espera a que el SoC baje de esta temperatura.
        "cooldown_temp_c": 55.0,
        "cooldown_timeout_s": 300,
        "cooldown_poll_s": 5,
    },

    # 7. Configuración dinámica de Salidas y Reportes
    "output_settings": {
        "base_dir": os.path.join(BASE_DIR, "outputs"),
        "subfolders": ['frames', 'reports', 'graphs'],
        "report_name": "metrics_report.txt",
        "graph_name": "fps_comparison",
        "plot_dpi": 300,
        "plot_size": (10, 6)
    }
}
