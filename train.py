import cv2
import numpy as np
import os

dataset_path = "dataset"
faces = []
labels = []
label_map = {}
current_label = 0

for person in os.listdir(dataset_path):
    label_map[current_label] = person
    person_path = os.path.join(dataset_path, person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(current_label)

    current_label += 1

faces = np.array(faces)
labels = np.array(labels)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)
model.save("face_model.yml")

print("Training completed")