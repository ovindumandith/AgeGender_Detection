import cv2
import numpy as np

class AgeGenderPredictor:
    def __init__(self, age_proto_path, age_model_path, 
                 gender_proto_path, gender_model_path):
        """
        Initialize the age and gender predictor with model paths
        
        Args:
            age_proto_path: Path to the age model prototxt file
            age_model_path: Path to the age model weights file
            gender_proto_path: Path to the gender model prototxt file
            gender_model_path: Path to the gender model weights file
        """
        # Initialize age network
        self.age_net = cv2.dnn.readNet(age_model_path, age_proto_path)
        
        # Initialize gender network
        self.gender_net = cv2.dnn.readNet(gender_model_path, gender_proto_path)
        
        # Age ranges
        self.age_ranges = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
        
        # Gender classes
        self.gender_list = ['Male', 'Female']

    def predict(self, frame, face_box):
        """
        Predict age and gender for a detected face
        
        Args:
            frame: Original image frame
            face_box: Bounding box coordinates (x, y, w, h)
            
        Returns:
            Predicted age range and gender
        """
        # Extract face from the frame
        x, y, w, h = face_box
        face = frame[y:y+h, x:x+w]
        
        # Check if face is valid
        if face.size == 0:
            return "Unknown", "Unknown"
        
        # Prepare blob for gender prediction
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                    (78.4263377603, 87.7689143744, 114.895847746),
                                    swapRB=False)
        
        # Gender prediction
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]
        
        # Prepare blob for age prediction
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.age_ranges[age_preds[0].argmax()]
        
        return age, gender