import cv2
import numpy as np

def run_video_filter_demo(video_path="rover_feed.mp4"):
    print("[SYSTEM] Initializing Environmental Preprocessing Demo...")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Could not load {video_path}. Make sure it is in the same folder!")
        return

    # Initialize the CLAHE algorithm once outside the loop for speed
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    try:
        while True:
            ret, frame = cap.read()

            # --- AUTO-LOOP ---
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            # Resize individual frames so the side-by-side fits on a laptop screen
            frame = cv2.resize(frame, (600, 450))

            # --- APPLY CLAHE FILTER ---
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # --- DRAW PRESENTATION LABELS ---
            cv2.putText(frame, "RAW SENSOR FEED", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(enhanced_frame, "CLAHE ANTI-GLARE", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # --- STITCH AND DISPLAY ---
            comparison = np.hstack((frame, enhanced_frame))
            cv2.imshow("LEFT: Raw Lunar Feed  |  RIGHT: CLAHE Edge Optimization", comparison)

            # Wait 30ms for standard framerate, exit on 'q'
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[OPERATOR OVERRIDE] Manual shutdown signal received.")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Filter Demo offline.")

if __name__ == "__main__":
    run_video_filter_demo()
    