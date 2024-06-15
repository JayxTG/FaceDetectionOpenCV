import cv2
import sys
import numpy as np
import os
import tkinter as tk
from tkinter import simpledialog

# Haarcascade file for face detection
haar_file = 'haarcascade_frontalface_default.xml'

# Function to prompt for the name using tkinter
def get_name():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    user_name = simpledialog.askstring(title="Name Prompt", prompt="Enter the name of the person:")
    root.destroy()  # Destroy the main window after getting the input
    return user_name

# Prompt for the name of the person
sub_data = get_name()

# Check if a name was provided
if not sub_data:
    print("Name not provided, exiting...")
    sys.exit()

# Folder to store the captured face images
datasets = 'datasets'

# Create the dataset folder if it doesn't exist
path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.makedirs(path)

# Define the size of the images
(width, height) = (130, 100)

# Load the Haarcascade file
face_cascade = cv2.CascadeClassifier(haar_file)

# Initialize the webcam
webcam = cv2.VideoCapture(0)

# Initialize the image counter
count = 1

# The program loops until it has 300 images of the face
while count <= 300:
    # Capture frame-by-frame
    ret, im = webcam.read()
    
    if not ret:
        print("Failed to capture image")
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Extract the face from the grayscale image
        face = gray[y:y + h, x:x + w]
        
        # Resize the face image to the defined size
        face_resize = cv2.resize(face, (width, height))
        
        # Save the face image to the dataset folder
        cv2.imwrite(f'{path}/{count}.png', face_resize)
        
        # Increment the image counter
        count += 1
    
    # Display the frame with the face rectangle
    cv2.imshow('OpenCV', im)
    
    # Exit the loop if the 'Esc' key is pressed
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
