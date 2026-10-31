import os
from config import CONFIG
from modules.inferencer import run_inference
from modules.analyzer import extract_significant_frames
from modules.reporter import generate_reports_and_graphs
from modules import sysmon

def create_dirs(base_dir, subfolders):
    for sub in subfolders:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

if __name__ == "__main__":
    print("Iniciando orquestación de datos...")

    out_settings = CONFIG['output_settings']
    create_dirs(out_settings['base_dir'], out_settings['subfolders'])

    modes = CONFIG['modes']                                       # ['detect', 'track']
    model_sizes = list(CONFIG['models'].keys())                   # ['Nano', 'Small', 'Medium']
    format_names = list(CONFIG['models'][model_sizes[0]].keys())  # ['pytorch', 'onnx', 'tflite']
    thermal = CONFIG['thermal']

    # 1. Estructura: all_metrics[modo][tamaño][formato] = lista de métricas por frame
    all_metrics = {mode: {size: {} for size in model_sizes} for mode in modes}

    # 2. Bucle por MODO (detect -> track) -> FORMATO -> TAMAÑO
    for mode in modes:
        print(f"\n{'#'*50}")
        print(f" MODO: {mode.upper()}")
        print(f"{'#'*50}")

        # En modo track se infiere con un umbral menor para que ByteTrack reciba
        # también las detecciones de baja confianza (2a asociación).
        conf = CONFIG['track_conf_threshold'] if mode == "track" else CONFIG['conf_threshold']

        for format_name in format_names:
            print(f"\n{'='*40}")
            print(f" INICIANDO FORMATO: {format_name.upper()} ({mode})")
            print(f"{'='*40}")

            for model_size in model_sizes:
                model_path = CONFIG['models'][model_size][format_name]
                combined_name = f"{model_size}_{format_name}_{mode}"
                print(f"\nEvaluando: {combined_name}...")

                # Cada corrida parte de una temperatura comparable
                waited, temp = sysmon.wait_for_cooldown(
                    thermal['cooldown_temp_c'],
                    thermal['cooldown_timeout_s'],
                    thermal['cooldown_poll_s'])
                if temp is not None:
                    print(f"   Temp. inicial: {temp:.1f} °C (espera de enfriamiento: {waited:.0f} s)")

                try:
                    metrics, best_frame = run_inference(
                        model_path,
                        CONFIG['video_path'],
                        conf,
                        CONFIG['img_size'],
                        mode=mode,
                        tracker_cfg=CONFIG['tracker_cfg'],
                        warmup_frames=CONFIG['warmup_frames'],
                        sample_every=CONFIG['sample_every'],
                    )

                    all_metrics[mode][model_size][format_name] = metrics
                    extract_significant_frames(best_frame, combined_name, out_settings['base_dir'])

                    if metrics:
                        mean_lat = sum(m['latency'] for m in metrics) / len(metrics)
                        temps = [m['temp_c'] for m in metrics if m['temp_c'] is not None]
                        tmax = f", Tmax {max(temps):.1f} °C" if temps else ""
                        print(f"   -> {1000/mean_lat:.2f} FPS ({mean_lat:.1f} ms){tmax}")

                except Exception as e:
                    print(f"-> Omitiendo {combined_name}. Error: {e}")
                    all_metrics[mode][model_size][format_name] = []

    print("\nGenerando reportes y gráficas...")
    generate_reports_and_graphs(all_metrics, out_settings)
    print(f"Evaluación completada exitosamente. Resultados en: {out_settings['base_dir']}")
