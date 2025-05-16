import cv2

def draw_predictions(frame, face_box, age, gender):
    """
    Draw bounding box and predictions on the frame
    
    Args:
        frame: Original image frame
        face_box: Face bounding box (x, y, w, h)
        age: Predicted age range
        gender: Predicted gender
        
    Returns:
        Frame with annotations
    """
    x, y, w, h = face_box
    
    # Draw rectangle around face
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Add labels
    label = f"{gender}, {age}"
    y_pos = y - 15 if y - 15 > 15 else y + 15
    cv2.putText(frame, label, (x, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame