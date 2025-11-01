import cv2
import os
import numpy as np

def build_gabor_kernels():
    kernels = []
    ksize = 31  # kernel size
    for theta in np.arange(0, np.pi, np.pi/4):
        for sigma in (4.0, 8.0):
            for lamda in np.arange(np.pi/4, np.pi, np.pi/4):
                kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
                kernels.append(kern)
    return kernels

def apply_gabor(img, kernels):
    accum = np.zeros_like(img)
    for kern in kernels:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
gabor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_gabor")
os.makedirs(gabor_dir, exist_ok=True)

kernels = build_gabor_kernels()

# Process each person's images
for person in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_path):
        continue
    save_path = os.path.join(gabor_dir, person)
    os.makedirs(save_path, exist_ok=True)
    
    for file in os.listdir(person_path):
        img_path = os.path.join(person_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        gabor_img = apply_gabor(img, kernels)
        cv2.imwrite(os.path.join(save_path, file), gabor_img)

print("Gabor preprocessing completed!")
