import cv2
import json
import time
from src.perception import AILunarPerception

def run_rover_core():
    print("[SYSTEM] Initializing C.R.A.T.E.R. Edge AI Core...")
    
    # 1. Initialize the AI with the highly-curated 500-image model
    vision_system = AILunarPerception(
        model_path=r"src\lunar_model_v1_500.pt\best.pt", 
        confidence_threshold=0.35
    )
    
    # 2. Setup Telemetry Logging
    log_file = open("mission_telemetry.jsonl", "a")
    
    # 3. Load the Video Feed
    video_path = "rover_feed.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not load {video_path}. Make sure it is in the same folder as main.py!")
        return

    print("[SYSTEM] Video feed acquired. Commencing live inference...")

    try:
        while True:
            ret, frame = cap.read()
            
            # If the video finishes, loop back to the start
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            
            # Perception pipeline
            perception_data = vision_system.process_frame(frame)
            
            # Draw detections
            if perception_data["hazard_count"] > 0:
                print(f"[TELEMETRY] -> {perception_data}")
                for hazard in perception_data["hazards"]:
                    x, y, w, h = hazard["bbox"]
                    conf = hazard["confidence"]
                    name = hazard["type"]
                    label = f"{name} {conf}"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            cv2.imshow("Rover NavCam - Live Video Feed", frame)
            
            # Press 'q' while focused on the video window to quit cleanly
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[OPERATOR OVERRIDE] Manual shutdown signal received.")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()
        print("[SYSTEM] Rover Core offline. Telemetry stream saved.")

if __name__ == "__main__":
    run_rover_core()