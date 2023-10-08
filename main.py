import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load Haar cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Load emotion recognition model
emotion_model = load_model('models/emotion_model.h5')

# Read the input image
img = cv2.imread('test.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    # Extract detected face
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48))  # resize to model input size
    roi_gray = np.expand_dims(roi_gray, axis=0)  # add batch dimension
    roi_gray = np.expand_dims(roi_gray, axis=-1)  # add channel dimension

    # Predict emotion
    emotion = emotion_model.predict(roi_gray)
    emotion_label = np.argmax(emotion)

    # Draw rectangle around face and label with predicted emotion
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the output image
cv2.imshow('Emotion Recognition', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
