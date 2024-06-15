# Face Recognition with OpenCV

## Project Description

This project implements a face recognition system using OpenCV in Python. It involves creating a dataset of faces and recognizing them in real-time using a webcam. The project uses the `haarcascade_frontalface_default.xml` file for detecting faces and the `face_recognize.py` and `create_data.py` scripts for creating and recognizing face data.

## 🎯 Objective

The objective of this project is to develop a face recognition system capable of detecting and recognizing faces in real-time, leveraging the power of OpenCV's Haar Cascade Classifiers.

## 🔑 Key Features

- 📸 Real-time face detection and recognition
- 📂 Creation of a face dataset using a webcam
- 🖥️ Utilizes OpenCV's pre-trained Haar Cascade Classifier for face detection

## 🛠️ Hardware and Software Requirements

- **Hardware**: A computer with a webcam
- **Software**: 
  - Python
  - OpenCV

## 📁 Folder Structure

- `OpenCV Version check.py`: Python script to check the installed OpenCV version.
- `haarcascade_frontalface_default.xml`: Pre-trained model for face detection.
- `datasets`: Folder containing datasets used for training the face recognition model.\
 -- `face_recognize.py`: Script for recognizing faces.
 --`create_data.py`: Script for creating face data.
 --`haarcascade_frontalface_default.xml`: Pre-trained model for face detection.
- `Face Detection`: Folder containing files or resources related to face detection.
- `studysession`: Folder with an unclear purpose from the image.

## 🎛️ How to Use

### 1. Install Required Libraries

Ensure you have Python installed. Install the required libraries using pip:

```bash
pip install opencv-python
```

### 2. Create Face Data
Run the create_data.py script to capture and store face data from the webcam:

```bash
python create_data.py
```

### 3. Recognize Faces
Run the face_recognize.py script to recognize and label faces in real-time from the webcam:

```bash
python face_recognize.py
```

Ensure that the haarcascade_frontalface_default.xml file is in the same directory as the scripts.

📸 Sample Usage
Creating Face Data
The create_data.py script captures images from the webcam and stores them as face data for recognition. This script will guide you through the process of creating a dataset of faces.

Recognizing Faces
The face_recognize.py script uses the created dataset to recognize and label faces in real-time. This script will display the webcam feed with recognized faces labeled accordingly.

Dependencies
Python
OpenCV
