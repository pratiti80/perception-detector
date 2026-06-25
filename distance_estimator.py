import cv2
from ultralytics import YOLO
import time

# Load a lightweight, fast model
model = YOLO("yolov8n.pt") 

# Constants
REAL_CAR_WIDTH = 1.8  # meters
FOCAL_LENGTH = 700.0  # pixels

# Open video file (e.g., a clip from KITTI dataset)
cap = cv2.VideoCapture("data/test.mp4")

past_distances = {}
smooth_distances = {}
prev_time = time.time()

while cap.isOpened():
    current_time = time.time()
    delta_t = current_time - prev_time
    prev_time = current_time
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference (classes=[2, 7] targets cars and trucks in COCO)
    results = model.track(frame, classes=[2, 7], persist=True, verbose=False, tracker='botsort.yaml')

    
    for box in results[0].boxes:
            # Get 2D bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            if bbox_width > 0:
                # Calculate distance 
                ALPHA = 0.3
                track_id = int(box.id[0]) if box.id is not None else -1
                raw_dist = (REAL_CAR_WIDTH * FOCAL_LENGTH) / bbox_width
                if track_id in smooth_distances:
                    smooth_distances[track_id] = ALPHA * raw_dist + (1 - ALPHA) * smooth_distances[track_id]
                else:
                    smooth_distances[track_id] = raw_dist
                
                distance = smooth_distances[track_id]

                ttc = float('inf')
                if track_id in past_distances:
                    rel_speed = (past_distances[track_id] - distance) / delta_t
                    if rel_speed > 0:
                        ttc = distance / rel_speed
                past_distances[track_id] = distance
                        

                # Calculate perspective shift for the 3D footprint
                # As objects get further away, the simulated 3D depth lines should look shorter
                box_depth_pixels = int((bbox_width) * 0.5) 
                skew = int(bbox_width * 0.1) # Gives it a slight angle relative to the camera

                # Define the 4 bottom corners of the 3D box on the road
                front_left  = (x1, y2)
                front_right = (x2, y2)
                back_left   = (x1 + skew, y2 - box_depth_pixels)
                back_right  = (x2 + skew, y2 - box_depth_pixels)

                # Define the 4 top corners of the 3D box
                top_front_left  = (x1, y1)
                top_front_right = (x2, y1)
                top_back_left   = (x1 + skew, y1 - box_depth_pixels)
                top_back_right  = (x2 + skew, y1 - box_depth_pixels)

                # --- DRAWING THE 3D BOX ---
                if ttc == float('inf') or ttc > 5:
                    color = (0, 255, 0) 
                elif 2.5 < ttc <= 5:
                    color = (0, 165, 255)
                elif ttc <= 2.5:
                    color = (0, 0, 255)
                    
                    
                thickness = 2

                # Draw Base/Ground Plane
                cv2.line(frame, front_left, front_right, color, thickness)
                cv2.line(frame, front_right, back_right, color, thickness)
                cv2.line(frame, back_right, back_left, color, thickness)
                cv2.line(frame, back_left, front_left, color, thickness)

                # Draw Top Plane
                cv2.line(frame, top_front_left, top_front_right, color, thickness)
                cv2.line(frame, top_front_right, top_back_right, color, thickness)
                cv2.line(frame, top_back_right, top_back_left, color, thickness)
                cv2.line(frame, top_back_left, top_front_left, color, thickness)

                # Draw Vertical Pillars connecting top and bottom
                cv2.line(frame, front_left, top_front_left, color, thickness)
                cv2.line(frame, front_right, top_front_right, color, thickness)
                cv2.line(frame, back_left, top_back_left, color, thickness)
                cv2.line(frame, back_right, top_back_right, color, thickness)
                

                # HUD Overlay: Print id and distance on top of the 3D box
                cv2.putText(frame, f"Z: {distance:.1f}m", (x1, y1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
                if ttc != float('inf') and ttc <= 5:
                    ttc_text = f"TTC: {ttc:.1f}s"
                    cv2.putText(frame, ttc_text, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Perception MVP", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()