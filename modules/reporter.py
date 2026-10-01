import matplotlib.pyplot as plt
import numpy as np
import os

def generate_reports_and_graphs(all_metrics, out_settings):
    base_dir = out_settings['base_dir']
    report_path = os.path.join(base_dir, "reports", out_settings['report_name'])
    
    # Extraer las claves (Nano, Small, Medium)
    sizes = list(all_metrics.keys())
    # Extraer los formatos (pytorch, onnx, tflite) asumiendo que el primer modelo los tiene todos
    formats = list(all_metrics[sizes[0]].keys())
    
    # Diccionario para almacenar los FPS por formato para graficar
    fps_data = {fmt: [] for fmt in formats}
    
    # --- Generación del Reporte de Texto ---
    with open(report_path, 'w') as f:
        f.write("Reporte de Evaluación por Versión y Formato\n")
        f.write("="*50 + "\n\n")
        
        for size in sizes:
            f.write(f"--- VERSIÓN: {size.upper()} ---\n")
            for fmt in formats:
                metrics = all_metrics[size].get(fmt)
                if not metrics:
                    f.write(f"  Formato {fmt}: Sin datos.\n")
                    fps_data[fmt].append(0)
                    continue
                
                lats = [m['latency'] for m in metrics]
                mean_lat = np.mean(lats) if lats else 0
                fps = 1000 / mean_lat if mean_lat > 0 else 0
                mean_ram = np.mean([m['ram'] for m in metrics]) if metrics else 0
                
                # Guardar FPS para la gráfica
                fps_data[fmt].append(fps)
                
                f.write(f"  Formato: {fmt}\n")
                f.write(f"    Latencia Media: {mean_lat:.2f} ms\n")
                f.write(f"    FPS Promedio:   {fps:.2f}\n")
                f.write(f"    RAM Promedio:   {mean_ram:.2f} MB\n")
            f.write("\n")

    # --- Generación de Gráficas Agrupadas ---
    x = np.arange(len(sizes))  # Ubicaciones de las etiquetas en X (Nano, Small, Medium)
    width = 0.25  # Ancho de las barras
    multiplier = 0

    # Usamos plot_size de la configuración
    fig, ax = plt.subplots(figsize=out_settings['plot_size'], layout='constrained')

    for attribute, measurement in fps_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fmt='%.1f')
        multiplier += 1

    ax.set_ylabel('Cuadros por Segundo (FPS)')
    ax.set_title('Comparativa de FPS por Tamaño de Modelo y Formato')
    ax.set_xticks(x + width, sizes)
    ax.legend(loc='upper left', ncols=3)
    
    # Ajustar límite Y dinámicamente para que no corte los números superiores
    max_fps = max([max(v) for v in fps_data.values()]) if fps_data else 0
    ax.set_ylim(0, max_fps * 1.2)

    # Guardado usando variables de configuración
    base_graph_path = os.path.join(base_dir, "graphs", out_settings['graph_name'])
    plt.savefig(f"{base_graph_path}.png", dpi=out_settings['plot_dpi'])
    plt.savefig(f"{base_graph_path}.svg", format='svg')
    plt.close()