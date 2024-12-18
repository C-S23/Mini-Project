import cv2
import csv
import os
import time
from datetime import datetime
from screeninfo import get_monitors
from deepface import DeepFace

class FaceRecognitionSystem:
    def __init__(self):
        self.reference_images = {}
        self.enrollment_dir = "enrolled_faces"
        if not os.path.exists(self.enrollment_dir):
            os.makedirs(self.enrollment_dir)
        if not os.path.exists("Attendance"):
            os.makedirs("Attendance")
        for filename in os.listdir(self.enrollment_dir):
         if filename.endswith('.jpg'):
            name = filename[:-4]  # Remove .jpg extension
            image_path = os.path.join(self.enrollment_dir, filename)
            self.reference_images[name] = image_path
            
    def enroll_person(self, name, face_image):
        image_path = os.path.join(self.enrollment_dir, f"{name}.jpg")
        cv2.imwrite(image_path, face_image)
        self.reference_images[name] = image_path
        return True

def main():
    system = FaceRecognitionSystem()
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = video.read()
        if not ret:
            continue

        faces = facedetect.detectMultiScale(frame, 1.3, 5)
        
        # Handle enrollment (press 'e')
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e') and len(faces) > 0:
            name = input("Enter person's name: ")
            x, y, w, h = faces[0]
            face_image = frame[y:y+h, x:x+w]
            if system.enroll_person(name, face_image):
                print(f"Successfully enrolled {name}")

        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            person_name = "Unknown"
            
            # Verify against enrolled faces
            for name, ref_img in system.reference_images.items():
                try:
                    result = DeepFace.verify(face_region, ref_img, enforce_detection=False)
                    if result["verified"]:
                        person_name = name
                        # Log attendance
                        date = datetime.now().strftime("%Y-%m-%d")
                        time_str = datetime.now().strftime("%H:%M:%S")
                        attendance_file = f"Attendance/attendance_{date}.csv"
                        
                        with open(attendance_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            if os.path.getsize(attendance_file) == 0:
                                writer.writerow(['NAME', 'TIME', 'DATE'])
                            writer.writerow([name, time_str, date])
                        break
                except:
                    continue

            cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
            cv2.putText(frame, person_name, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

