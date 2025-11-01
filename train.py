import cv2
import os
import numpy as np

dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_gabor")

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
        faces.append(img)
        labels.append(label_id)
    label_id += 1

faces = [cv2.resize(f, (100, 100)) for f in faces]  # standard size
faces = np.array(faces)
labels = np.array(labels)

# Train LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

# Save model and label dictionary
recognizer.save("face_model.yml")
np.save("labels.npy", label_dict)

print("Model training completed!")
