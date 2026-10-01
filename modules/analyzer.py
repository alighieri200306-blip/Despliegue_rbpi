import cv2
import os

def extract_significant_frames(best_frame, model_name, output_dir):
    """Guarda directamente el fotograma más significativo si existe."""
    if best_frame is not None:
        frame_path = os.path.join(output_dir, "frames", f"{model_name}_best_frame.jpg")
        cv2.imwrite(frame_path, best_frame)