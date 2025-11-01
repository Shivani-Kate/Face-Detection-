
import cv2
import os
import numpy as np


def apply_gabor(img_gray):
    # Gabor parameters
    ksize = 31      # filter size
    sigma = 4.0     # standard deviation
    theta = np.pi/4 # orientation
    lamda = 10.0    # wavelength
    gamma = 0.5     # aspect ratio
    phi = 0

    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
    filtered = cv2.filter2D(img_gray, cv2.CV_8UC1, kernel)
    return filtered

dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")

faces = []
labels = []
label_dict = {}
label_id = 0

for person in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_path):
        continue
    label_dict[label_id] = person
    for file in os.listdir(person_path):
        img_path = os.path.join(person_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (100, 100))        # Resize to fix array shape
        img = apply_gabor(img)                  
        faces.append(img)
        labels.append(label_id)
    label_id += 1

faces = np.array(faces)
labels = np.array(labels)


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
print("Real-time recognition started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        face_img = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        face_img = apply_gabor(face_img)  # Gabor preprocessing
        label, conf = recognizer.predict(face_img)
        name = label_dict.get(label, "Unknown")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({int(conf)})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
