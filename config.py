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
    
    # 4. Configuración dinámica de Salidas y Reportes
    "output_settings": {
        "base_dir": os.path.join(BASE_DIR, "outputs"),
        "subfolders": ['frames', 'reports', 'graphs'],
        "report_name": "metrics_report.txt",
        "graph_name": "fps_comparison",
        "plot_dpi": 300,
        "plot_size": (10, 6)
    }
}