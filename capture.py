import cv2
import os
import argparse

def capture_faces(name, count=50):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    person_dir = os.path.join(dataset_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    print(f"Capturing {count} images for {name}. Press 'q' to quit early.")
    img_counter = 0

    while img_counter < count:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (100, 100))
            cv2.imwrite(os.path.join(person_dir, f"{name}_{img_counter+1}.jpg"), face_img)
            img_counter += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

        cv2.imshow("Capture Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {img_counter} images for {name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    capture_faces(args.name, args.count)
