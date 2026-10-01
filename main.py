import os
from config import CONFIG
from modules.inferencer import run_inference
from modules.analyzer import extract_significant_frames
from modules.reporter import generate_reports_and_graphs

def create_dirs(base_dir, subfolders):
    for sub in subfolders:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

if __name__ == "__main__":
    print("Iniciando orquestación de datos...")
    
    out_settings = CONFIG['output_settings']
    create_dirs(out_settings['base_dir'], out_settings['subfolders'])
    
    # 1. Pre-inicializar la estructura para evitar errores al invertir el orden
    all_model_metrics = {} 
    model_sizes = list(CONFIG['models'].keys())          # ['Nano', 'Small', 'Medium']
    format_names = list(CONFIG['models'][model_sizes[0]].keys()) # ['pytorch', 'onnx', 'tflite']
    
    for size in model_sizes:
        all_model_metrics[size] = {}

    # 2. NUEVO ORDEN: Bucle principal por FORMATO (pytorch -> onnx -> tflite)
    for format_name in format_names:
        print(f"\n{'='*40}")
        print(f" INICIANDO FORMATO: {format_name.upper()}")
        print(f"{'='*40}")
        
        # Bucle secundario por TAMAÑO (Nano -> Small -> Medium)
        for model_size in model_sizes:
            model_path = CONFIG['models'][model_size][format_name]
            combined_name = f"{model_size}_{format_name}" 
            print(f"\nEvaluando: {combined_name}...")
            
            try:
                # Desempaqueta la tupla devuelta por el inferidor optimizado
                metrics, best_frame = run_inference(
                    model_path, 
                    CONFIG['video_path'], 
                    CONFIG['conf_threshold'],
                    CONFIG['img_size']
                )
                
                # Almacena en la estructura original para que el reporte no se rompa
                all_model_metrics[model_size][format_name] = metrics
                
                # Guarda la imagen
                extract_significant_frames(best_frame, combined_name, out_settings['base_dir'])
                
            except Exception as e:
                print(f"-> Omitiendo {combined_name}. Error: {e}")
                all_model_metrics[model_size][format_name] = [] 

    print("\nGenerando reportes y gráficas...")
    generate_reports_and_graphs(all_model_metrics, out_settings)
    print(f"Evaluación completada exitosamente. Resultados en: {out_settings['base_dir']}")