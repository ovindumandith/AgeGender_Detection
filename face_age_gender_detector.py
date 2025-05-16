import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import time

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Press 'q' to quit")
    
    # For FPS calculation
    prev_frame_time = 0
    new_frame_time = 0
    
    # MediaPipe Face Detection
    with mp_face_detection.FaceDetection(
        model_selection=1,  # 0 for short-range, 1 for full-range detection
        min_detection_confidence=0.5) as face_detection:
        
        while cap.isOpened():
            # Read frame
            success, image = cap.read()
            if not success:
                print("Error: Failed to capture frame.")
                break
            
            # FPS calculation
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time
            fps_text = f"FPS: {int(fps)}"
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # To improve performance, mark image as not writeable
            image.flags.writeable = False
            
            # Process the image
            results = face_detection.process(image_rgb)
            
            # Allow image modifications again
            image.flags.writeable = True
            
            # Draw face detections and predict age/gender
            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                 int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    # Ensure coordinates are within image boundaries
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, iw - x)
                    h = min(h, ih - y)
                    
                    # Extract face
                    face_img = image[y:y+h, x:x+w]
                    
                    # Only process if face is valid
                    if face_img.size > 0:
                        try:
                            # Predict age and gender
                            # We'll use deepface for this (note: first run will download models)
                            result = DeepFace.analyze(
                                face_img, 
                                actions=['age', 'gender'],
                                enforce_detection=False,
                                silent=True
                            )
                            
                            # Get prediction results
                            if isinstance(result, list):
                                result = result[0]  # Get first face if multiple detected
                                
                            age = result['age']
                            gender = result['gender']
                            gender_confidence = result['gender_confidence']
                            
                            # Format gender with confidence
                            gender_label = f"{gender} ({gender_confidence:.2f})"
                            
                            # Draw bounding box
                            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            
                            # Add labels
                            label1 = f"Age: {age}"
                            label2 = f"Gender: {gender_label}"
                            
                            # Position labels
                            cv2.putText(image, label1, (x, y-35 if y-35 > 15 else y+h+25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(image, label2, (x, y-10 if y-10 > 15 else y+h+50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                        except Exception as e:
                            # Draw bounding box but without prediction
                            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            print(f"Error in prediction: {e}")
            
            # Add FPS counter
            cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Display the result
            cv2.imshow('Age and Gender Detection', image)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()