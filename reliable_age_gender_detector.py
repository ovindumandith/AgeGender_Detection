import cv2 as cv
import time
import os
import numpy as np

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

def main():
    # Model file paths - adjust these to point to your models folder
    models_dir = "models"
    faceProto = os.path.join(models_dir, "opencv_face_detector.pbtxt")
    faceModel = os.path.join(models_dir, "opencv_face_detector_uint8.pb")
    ageProto = os.path.join(models_dir, "age_deploy.prototxt")
    ageModel = os.path.join(models_dir, "age_net.caffemodel")
    genderProto = os.path.join(models_dir, "gender_deploy.prototxt")
    genderModel = os.path.join(models_dir, "gender_net.caffemodel")
    
    # Check if model files exist
    missing_files = []
    for model_file in [faceProto, faceModel, ageProto, ageModel, genderProto, genderModel]:
        if not os.path.exists(model_file):
            missing_files.append(model_file)
    
    if missing_files:
        print("The following model files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure these files exist in your 'models' directory.")
        
        # As a fallback, let's create the prototxt files if they're missing
        for file in missing_files:
            if file.endswith('.prototxt'):
                os.makedirs(os.path.dirname(file), exist_ok=True)
                with open(file, 'w') as f:
                    if "age" in file:
                        f.write("""
name: "AgeNet"
input: "data"
input_dim: 1
input_dim: 3
input_dim: 227
input_dim: 227

layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 96
    kernel_size: 7
    stride: 4
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "norm1"
  type: "LRN"
  bottom: "pool1"
  top: "norm1"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "fc8-101"
  type: "InnerProduct"
  bottom: "norm1"
  top: "fc8-101"
  inner_product_param {
    num_output: 8
  }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "fc8-101"
  top: "prob"
}
                        """)
                    elif "gender" in file:
                        f.write("""
name: "GenderNet"
input: "data"
input_dim: 1
input_dim: 3
input_dim: 227
input_dim: 227

layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 96
    kernel_size: 7
    stride: 4
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "norm1"
  type: "LRN"
  bottom: "pool1"
  top: "norm1"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "fc8-101"
  type: "InnerProduct"
  bottom: "norm1"
  top: "fc8-101"
  inner_product_param {
    num_output: 2
  }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "fc8-101"
  top: "prob"
}
                        """)
                print(f"Created {file}")
    
    # Model parameters
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
    
    # Load networks
    try:
        print("Loading face detection model...")
        faceNet = cv.dnn.readNet(faceModel, faceProto)
        
        # Try to load from direct file path if above fails
        if not os.path.exists(faceModel) or not os.path.exists(faceProto):
            alternate_face_model = "opencv_face_detector_uint8.pb"
            alternate_face_proto = "opencv_face_detector.pbtxt"
            if os.path.exists(alternate_face_model) and os.path.exists(alternate_face_proto):
                print("Using alternative face model path...")
                faceNet = cv.dnn.readNet(alternate_face_model, alternate_face_proto)
        
        print("Loading age prediction model...")
        ageNet = cv.dnn.readNet(ageModel, ageProto)
        
        # Try alternate paths if fails
        if not os.path.exists(ageModel) or not os.path.exists(ageProto):
            alternate_age_model = "age_net.caffemodel"
            alternate_age_proto = "age_deploy.prototxt"
            if os.path.exists(alternate_age_model) and os.path.exists(alternate_age_proto):
                print("Using alternative age model path...")
                ageNet = cv.dnn.readNet(alternate_age_model, alternate_age_proto)
        
        print("Loading gender prediction model...")
        genderNet = cv.dnn.readNet(genderModel, genderProto)
        
        # Try alternate paths if fails
        if not os.path.exists(genderModel) or not os.path.exists(genderProto):
            alternate_gender_model = "gender_net.caffemodel"
            alternate_gender_proto = "gender_deploy.prototxt"
            if os.path.exists(alternate_gender_model) and os.path.exists(alternate_gender_proto):
                print("Using alternative gender model path...")
                genderNet = cv.dnn.readNet(alternate_gender_model, alternate_gender_proto)
        
        # Set to CPU
        faceNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
        ageNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
        genderNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
        
        print("All models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Open webcam
    print("Opening webcam...")
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Press Q to quit")
    padding = 20
    
    # For FPS calculation
    prev_frame_time = 0
    new_frame_time = 0
    
    while True:
        # Read frame
        t = time.time()
        hasFrame, frame = cap.read()
        if not hasFrame:
            print("Error: Could not read frame.")
            break
        
        # FPS calculation
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {int(fps)}"
        
        # Detect faces
        frameFace, bboxes = getFaceBox(faceNet, frame)
        
        # Add FPS to display
        cv.putText(frameFace, fps_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2, cv.LINE_AA)
        
        if not bboxes:
            cv.putText(frameFace, "No Face Detected", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv.imshow("Age Gender Demo", frameFace)
            
            # Exit on 'q' key press
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue
        
        # Process each detected face
        for bbox in bboxes:
            try:
                # Extract face with padding
                face = frame[
                    max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1),
                    max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)
                ]
                
                # Skip if face is too small
                if face.size == 0 or face.shape[0] < 20 or face.shape[1] < 20:
                    continue
                
                # Create blob and predict gender
                blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                
                # Gender prediction
                try:
                    genderNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]
                    gender_conf = genderPreds[0].max() * 100
                except Exception as e:
                    print(f"Error in gender prediction: {e}")
                    gender = "Unknown"
                    gender_conf = 0
                
                # Age prediction
                try:
                    ageNet.setInput(blob)
                    agePreds = ageNet.forward()
                    age = ageList[agePreds[0].argmax()]
                    age_conf = agePreds[0].max() * 100
                except Exception as e:
                    print(f"Error in age prediction: {e}")
                    age = "Unknown"
                    age_conf = 0
                
                # Create label with prediction and confidence
                label = f"{gender} ({gender_conf:.1f}%), {age} ({age_conf:.1f}%)"
                
                # Add label to display
                cv.putText(frameFace, label, (bbox[0], bbox[1]-10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
                
            except Exception as e:
                print(f"Error processing face: {e}")
        
        # Display the result
        cv.imshow("Age Gender Demo", frameFace)
        
        # Processing time
        processing_time = time.time() - t
        if processing_time > 0.1:  # Only print if it's taking a significant amount of time
            print(f"Processing time: {processing_time:.3f} seconds")
        
        # Exit on 'q' key press
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv.destroyAllWindows()
    print("Application closed")

if __name__ == "__main__":
    main()