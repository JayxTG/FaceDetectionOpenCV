import cv2
import sys
import numpy as np
import os

# Constants
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
sub_data = 'jayamadu'
(width, height) = (130, 100)

# Part 1: Create LBPHFaceRecognizer
print('Recognizing Face. Please be in sufficient light...')

# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

# Create a Numpy array from the lists of images and labels
(images, labels) = [np.array(lis) for lis in [images, labels]]

# OpenCV trains a model from the images
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Part 2: Use LBPHFaceRecognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, im = webcam.read()
    if not ret:
        print("Failed to capture image")
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        
        # Try to recognize the face
        prediction = model.predict(face_resize)
        
        # Draw the rectangle around the face and put the label
        if prediction[1] < 500:
            cv2.putText(im, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        else:
            cv2.putText(im, 'Not Recognized', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    
    # Display the resulting frame
    cv2.imshow('OpenCV', im)
    
    # Exit the loop if the 'Esc' key is pressed
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
