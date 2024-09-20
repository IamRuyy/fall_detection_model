import cv2
import os
from ultralytics import YOLO
from datetime import datetime

# Initialize the YOLOv8 model
model = YOLO(r'C:\Users\deadn\Desktop\fall_detection\train10\weights\best.pt')

# Use the real-time camera feed (0 is for the default camera, change if you have multiple cameras)
video_path = r'C:\Users\deadn\Desktop\fall_detection\fall_5.mp4'
cap = cv2.VideoCapture(video_path)

# Create a directory to save the detected fall images
save_dir = r'C:\Users\deadn\Desktop\fall_detection\fall_detected'
os.makedirs(save_dir, exist_ok=True)

frame_count = 0
fall_start_time = None
fall_detected = False
images_captured = 0
fall_images_saved = 0  # Counter for all fall images saved
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera. Exiting...")
        break

    frame_count += 1
    current_time = frame_count / fps
    
    # Process every 5th frame to reduce computational load
    if frame_count % 5 == 0:
        results = model(frame)
        
        fall_in_current_frame = False
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0].item())  # Class ID (0 for fall, assuming your model)
                conf = box.conf[0].item()  # Confidence of detection
                
                if class_id == 0 and conf > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Fall detection logic: width > height (i.e., lying down)
                    if (x2 - x1) > (y2 - y1):
                        fall_in_current_frame = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f'Fall_Detected: {conf:.2f}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
                        # Set fall start time if not already set
                        if fall_start_time is None:
                            fall_start_time = current_time
                        
                        # Check if the person has been fallen for more than 10 seconds
                        elif current_time - fall_start_time >= 5 and not fall_detected:
                            fall_detected = True
                            images_captured = 0  # Reset image count for the new fall
                            print(f"Fall detected at frame {frame_count}, after {current_time - fall_start_time:.2f} seconds")
        
        # If no fall detected in the current frame, reset the fall timer
        if not fall_in_current_frame:
            fall_start_time = None
            fall_detected = False  # Reset fall detection if no fall in the frame
        
        # If a fall is detected, save exactly 2 images of the fall
        if fall_detected and images_captured < 2:
            fall_images_saved += 1
            
            # Create filename in the format 'fallen_person_1', 'fallen_person_2', etc.
            filename = f'fallen_person_{fall_images_saved}.jpg'
            save_path = os.path.join(save_dir, filename)
            
            # Save the image
            cv2.imwrite(save_path, frame)
            print(f"Fall image saved as {save_path}")
            images_captured += 1
        
        # After saving 2 images, exit the loop and stop the process
        if images_captured == 2:
            print("Saved 2 images of fallen person. Ending process.")
            break  # Exit the loop after saving 2 images
        
        # Display the frame
        cv2.imshow('Fall Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Total frames processed: {frame_count}")
print(f"Total fall images saved: {fall_images_saved}")
