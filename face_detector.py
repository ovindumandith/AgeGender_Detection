import cv2
import numpy as np

class FaceDetector:
    def __init__(self, prototxt_path, model_path, confidence_threshold=0.5):
        """
        Initialize the face detector with the model paths
        
        Args:
            prototxt_path: Path to the prototxt file
            model_path: Path to the model weights file
            confidence_threshold: Minimum confidence to consider a detection valid
        """
        self.confidence_threshold = confidence_threshold
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        
    def detect_faces(self, frame):
        """
        Detect faces in the input frame
        
        Args:
            frame: Input image frame from camera
            
        Returns:
            List of face bounding boxes (x, y, w, h) and confidence scores
        """
        # Get frame dimensions
        (h, w) = frame.shape[:2]
        
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        
        # Pass the blob through the network
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # List to store face locations
        face_boxes = []
        
        # Loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter weak detections
            if confidence > self.confidence_threshold:
                # Compute the (x, y)-coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Add face bounding box and confidence to results
                face_boxes.append({
                    "box": (startX, startY, endX - startX, endY - startY),
                    "confidence": float(confidence)
                })
        
        return face_boxes