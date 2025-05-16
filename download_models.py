# download_models.py
import os
import urllib.request

def download_file(url, file_path):
    """Download a file from a URL to a specified path"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if not os.path.exists(file_path):
        print(f"Downloading {os.path.basename(file_path)}...")
        urllib.request.urlretrieve(url, file_path)
        print(f"Downloaded {os.path.basename(file_path)}")
    else:
        print(f"{os.path.basename(file_path)} already exists.")

def main():
    # Create models directory
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Face detection model
    download_file(
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        os.path.join(model_dir, "res10_300x300.caffemodel")
    )
    download_file(
        "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt",
        os.path.join(model_dir, "deploy.prototxt")
    )
    
    # Age prediction model
    download_file(
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel",
        os.path.join(model_dir, "age_net.caffemodel")
    )
    download_file(
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/deploy_age.prototxt",
        os.path.join(model_dir, "age_deploy.prototxt")
    )
    
    # Gender prediction model
    download_file(
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel",
        os.path.join(model_dir, "gender_net.caffemodel")
    )
    download_file(
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/deploy_gender.prototxt",
        os.path.join(model_dir, "gender_deploy.prototxt")
    )

if __name__ == "__main__":
    main()