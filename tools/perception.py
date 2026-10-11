import cv2
import numpy as np
from typing import Dict, Any
from ultralytics import YOLO

class RobustLunarPerception:
    """Deterministic fallback pipeline using classical CV and morphological filtering."""
    def __init__(self, clip_limit=3.0, tile_grid=(8, 8), min_area=100, max_area=15000):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        self.min_area = min_area
        self.max_area = max_area

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced_gray = self.clahe.apply(gray)
        thresh = cv2.adaptiveThreshold(
            enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hazards = []
        
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + (w // 2), y + (h // 2)
                
                hazard_type = "LUNAR_ROCK" if circularity > 0.55 else "CRATER_DEPRESSION"
                hazards.append({
                    "id": idx,
                    "type": hazard_type,
                    "confidence": round(circularity, 2),
                    "bbox": [x, y, w, h],
                    "centroid": [cx, cy]
                })

        return {
            "status": "FALLBACK_SUCCESS",
            "hazard_count": len(hazards),
            "hazards": hazards,
            "pipeline": "CLASSICAL_CV_FALLBACK"
        }


class AILunarPerception:
    """Primary Edge-AI pipeline backed by your custom YOLOv8 model with automatic fallback."""
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, enable_fallback: bool = True):
        self.conf_thresh = confidence_threshold
        self.enable_fallback = enable_fallback
        self.fallback_detector = RobustLunarPerception()
        
        try:
            print(f"[SYSTEM] Loading Primary Neural Weights: {model_path}")
            self.model = YOLO(model_path)
            self.ai_operational = True
        except Exception as e:
            print(f"[WARNING] Neural model failed to initialize ({e}). Falling back to Classical CV.")
            self.ai_operational = False

    def process_frame(self, frame: np.ndarray) -> dict:
        if self.ai_operational:
            try:
                # --- ANTI-GLARE PREPROCESSING (CLAHE) ---
                # Convert to LAB color space to equalize only the Lightness channel
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to cut through sun glare and heavy shadows
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                
                # Merge back to BGR format
                limg = cv2.merge((cl, a, b))
                glare_reduced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                # ----------------------------------------

                # --- EDGE COMPUTE OPTIMIZATION ---
                # imgsz=320 drastically reduces RAM/CPU usage for Raspberry Pi deployment
                results = self.model(glare_reduced_frame, verbose=False, conf=self.conf_thresh, imgsz=320)
                
                hazards = []
                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        if conf >= self.conf_thresh:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            w = x2 - x1
                            h = y2 - y1
                            cx = x1 + (w // 2)
                            cy = y1 + (h // 2)

                            cls_id = int(box.cls[0])
                            class_name = self.model.names[cls_id].upper()

                            hazards.append({
                                "type": class_name,
                                "confidence": round(conf, 2),
                                "bbox": [x1, y1, w, h],
                                "centroid": [cx, cy]
                            })

                return {
                    "status": "AI_SUCCESS",
                    "hazard_count": len(hazards),
                    "hazards": hazards,
                    "pipeline": "EDGE_OPTIMIZED_YOLO"
                }

            except Exception as e:
                print(f"\n[CRITICAL FAULT] Neural inference crashed: {e}")
                print("[SYSTEM] Engaging deterministic fallback...")

        # Fallback if AI fails on the Pi
        if getattr(self, 'enable_fallback', False):
            return self.fallback_detector.process_frame(frame)
        return {"status": "FAULT", "hazard_count": 0, "hazards": [], "pipeline": "NONE"}