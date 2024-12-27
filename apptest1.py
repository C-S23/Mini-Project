import os
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.spatial.distance import cosine
import time
import datetime  # Import datetime here

class FaceRecognitionSystem:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1024x768")
        self.setup_gui_frames()
        self.initialize_recognition_system()
        self.recognition_active = False
        self.cap = None

    def setup_gui_frames(self):
        self.video_frame = ttk.Frame(self.root)
        self.control_frame = ttk.Frame(self.root)
        self.status_frame = ttk.Frame(self.root)
        self.video_frame.pack(pady=10)
        self.control_frame.pack(pady=5)
        self.status_frame.pack(pady=5, fill=tk.X)
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()
        self.enroll_btn = ttk.Button(self.control_frame, text="Enroll New Face", command=self.start_enrollment)
        self.start_btn = ttk.Button(self.control_frame, text="Start Recognition", command=self.start_recognition)
        self.stop_btn = ttk.Button(self.control_frame, text="Stop", command=self.stop_recognition)
        self.enroll_btn.pack(side=tk.LEFT, padx=5)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(self.status_frame, text="System Ready", font=('Helvetica', 10))
        self.status_label.pack(pady=5)
        self.accuracy_label = ttk.Label(self.status_frame, text="Accuracy: --", font=('Helvetica', 10))
        self.accuracy_label.pack(pady=5)

    def initialize_recognition_system(self):
        self.model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.enrollment_dir = "enrolled_faces"
        self.attendance_dir = "Attendance_Records"
        for directory in [self.enrollment_dir, self.attendance_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        self.known_faces = self.load_enrolled_faces()
        self.last_detection = {}
        self.detection_log = []
        self.accuracy_metrics = {'total': 0, 'correct': 0}

    def load_enrolled_faces(self):
        known_faces = {}
        for filename in os.listdir(self.enrollment_dir):
            if filename.endswith('.jpg'):
                name = filename.split('_sample_')[0]
                image_path = os.path.join(self.enrollment_dir, filename)
                img = cv2.imread(image_path)
                if img is not None:
                    features = self.extract_features(img)
                    known_faces[filename] = features
        return known_faces

    def extract_features(self, face_image):
        face_image = cv2.resize(face_image, (224, 224))
        face_image = img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        face_image = preprocess_input(face_image)
        features = self.model.predict(face_image, verbose=0)
        return features.flatten()

    def check_face_quality(self, face_image):
        brightness = np.mean(face_image)
        laplacian = cv2.Laplacian(face_image, cv2.CV_64F).var()
        height, width = face_image.shape[:2]
        quality_score = 0
        if 40 < brightness < 250: quality_score += 1
        if laplacian > 100: quality_score += 1
        if height > 100 and width > 100: quality_score += 1
        return quality_score >= 2

    def capture_session(self, video_capture, num_samples):
        face_samples = []
        sample_count = 0
        while sample_count < num_samples:
            ret, frame = video_capture.read()
            if not ret: continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_region = frame[y:y+h, x:x+w]
                if self.check_face_quality(face_region):
                    face_samples.append(face_region)
                    sample_count += 1
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Capturing: {sample_count}/{num_samples}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    time.sleep(0.2)
            cv2.imshow('Enrollment', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        return face_samples

    def enroll_person(self, name, video_capture):
        print(f"\nEnrolling {name}")
        total_samples = 100
        samples_per_session = 50
        print("\nSession 1: Please look straight at camera in current lighting")
        samples1 = self.capture_session(video_capture, samples_per_session)
        print("\nPlease move to a different location with different lighting")
        input("Press Enter when ready for second session...")
        print("\nSession 2: Please provide different face angles")
        samples2 = self.capture_session(video_capture, samples_per_session)
        all_samples = samples1 + samples2
        for idx, face in enumerate(all_samples):
            image_path = os.path.join(self.enrollment_dir, f"{name}_sample_{idx}.jpg")
            cv2.imwrite(image_path, face)
            features = self.extract_features(face)
            self.known_faces[f"{name}_sample_{idx}"] = features
        print(f"\nSuccessfully enrolled {name} with {len(all_samples)} samples")

    def verify_person(self, face_features):
        best_match = "Unknown"
        best_score = float('inf')
        is_new_person = True
        confidence_threshold = 0.95
        for filename, stored_features in self.known_faces.items():
            similarity = cosine(face_features, stored_features)
            if similarity < best_score:
                best_score = similarity
                best_match = filename.split('_sample_')[0]
                if similarity < confidence_threshold:
                    is_new_person = False
        self.detection_log.append({
            'predicted': best_match,
            'confidence': best_score,
            'timestamp': datetime.datetime.now()
        })
        if not is_new_person:
            self.update_attendance(best_match)
        return best_match, best_score, is_new_person
        #added 5-minute check to prevent multiple attendance being logged
    def update_attendance(self, person_name):
        current_time = datetime.datetime.now()
        if person_name in self.last_detection:
            time_diff = (current_time - self.last_detection[person_name]).total_seconds()
            if time_diff < 300:  # 5-minute time difference to prevent multiple checks
                return
        self.last_detection[person_name] = current_time
        current_date = current_time.strftime("%Y-%m-%d")
        current_time_str = current_time.strftime("%H:%M:%S")
        csv_path = os.path.join(self.attendance_dir, f"attendance_{current_date}.csv")
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=['Name', 'Time', 'Date', 'Status']).to_csv(csv_path, index=False)
        df = pd.read_csv(csv_path)
        new_record = pd.DataFrame([[person_name, current_time_str, current_date, "Present"]], columns=['Name', 'Time', 'Date', 'Status'])
        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(csv_path, index=False)

    def calculate_accuracy(self):
        total_detections = len(self.detection_log)
        correct_matches = sum(1 for detection in self.detection_log if detection['confidence'] < 0.38)
        accuracy = (correct_matches / total_detections * 100) if total_detections > 0 else 0
        return {
            'accuracy': accuracy,
            'total_samples': total_detections,
            'correct_matches': correct_matches
        }

    def start_enrollment(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
        name = simpledialog.askstring("Input", "Enter person's name:")
        if name:
            self.status_label.config(text=f"Enrolling {name}...")
            success = self.enroll_person(name, self.cap)
            if success:
                messagebox.showinfo("Success", f"Successfully enrolled {name}")
            self.status_label.config(text="System Ready")

    def start_recognition(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
        self.recognition_active = True
        self.status_label.config(text="Recognition Active")
        self.update_frame()

    def stop_recognition(self):
        self.recognition_active = False
        self.status_label.config(text="System Ready")
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_frame(self):
        if self.recognition_active and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                self.cap = cv2.VideoCapture(0)
                ret, frame = self.cap.read()
                if not ret:
                    messagebox.showerror("Error", "Failed to open camera")
                    self.stop_recognition()
                    return
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_region = frame[y:y+h, x:x+w]
                if self.check_face_quality(face_region):
                    features = self.extract_features(face_region)
                    person_name, confidence, is_new = self.verify_person(features)
                    color = (0, 0, 255) if is_new else (0, 255, 0)
                    status = "New Person" if is_new else person_name
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{status} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            if len(self.detection_log) > 0:
                metrics = self.calculate_accuracy()
                self.accuracy_label.config(text=f"Accuracy: {metrics['accuracy']:.2f}%")
            self.root.after(10, self.update_frame)

def main():
    app = FaceRecognitionSystem()
    app.root.mainloop()

if __name__ == "__main__":
    main()
