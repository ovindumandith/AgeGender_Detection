import cv2
import os
import time
from face_detector import FaceDetector
from age_gender_predictor import AgeGenderPredictor
from utils.helper import draw_predictions

def main():
    # Paths to model files
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    
    # Face detection model files
    face_proto = os.path.join(model_dir, "deploy.prototxt")
    face_model = os.path.join(model_dir, "res10_300x300.caffemodel")
    
    # Age prediction model files
    age_proto = os.path.join(model_dir, "age_deploy.prototxt")
    age_model = os.path.join(model_dir, "age_net.caffemodel")
    
    # Gender prediction model files
    gender_proto = os.path.join(model_dir, "gender_deploy.prototxt")
    gender_model = os.path.join(model_dir, "gender_net.caffemodel")
    
    # Initialize the face detector
    face_detector = FaceDetector(face_proto, face_model, confidence_threshold=0.5)
    
    # Initialize the age and gender predictor
    age_gender_predictor = AgeGenderPredictor(age_proto, age_model, gender_proto, gender_model)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Press 'q' to quit")
    
    # Process frames
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Detect faces
        face_boxes = face_detector.detect_faces(frame)
        
        # Process each detected face
        for face_data in face_boxes:
            face_box = face_data["box"]
            
            # Predict age and gender
            age, gender = age_gender_predictor.predict(frame, face_box)
            
            # Draw predictions on frame
            frame = draw_predictions(frame, face_box, age, gender)
        
        # Display the resulting frame
        cv2.imshow('Age and Gender Detection', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()