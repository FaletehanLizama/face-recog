"""
Main application entry point.
Orchestrates face recognition, emotion detection, and database.
"""

import cv2
from database import get_database
from face_recognizer import FaceRecognizer
from emotion_detector import EmotionDetector
import numpy as np  # Add if not present


def register_new_face(recognizer, face_data):
    """Prompt for name and register a new face."""
    name = input(f"\n📸 Register new face (ID: {face_data['face_id']}): Enter name: ").strip()
    if name:
        recognizer.register_new_face(name, face_data["encoding"])
        print(f"✅ Face registered as '{name}'")
    return name


def main():
    # 1. Connect to database and load known faces
    print("Connecting to database...")
    db = get_database()
    if db.connect():
        print("✅ Connected to Google Sheets")
    else:
        print("⚠️ Running in offline mode")

    known_faces = db.get_all_faces()
    print(f"📋 Loaded {len(known_faces)} known faces\n")

    # 2. Initialize modules
    recognizer = FaceRecognizer(known_faces=known_faces)
    emotion_detector = EmotionDetector(min_confidence=0.5)

    # 3. Start camera
    print("Starting camera...")
    print("Press 'r' on unknown face to register\n")
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("http://192.168.137.4:4747/remote")

    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS = less lag

    print(f"Camera opened: {cap.isOpened()}")  # ADD HERE

    if not cap.isOpened():  # AND THIS CHECK
        print("❌ Error: Cannot open camera. Is one connected?")
        return

    # Store unknown faces for potential registration
    unknown_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for faster face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 4. Detect faces using grayscale (faster processing)
        faces = recognizer.process_frame(gray_frame)
        unknown_faces = [f for f in faces if f["is_unknown"]]

        # 5. Detect emotions for each face (use color for better accuracy)
        for face in faces:
            top, right, bottom, left = face["bounding_box"]
            face_image = frame[top:bottom, left:right]  # Color for emotion
            emotion = emotion_detector.detect_emotion(face_image)
            face["emotion"] = emotion

            # 6. Update database for known faces
            if face["name"] and not face["is_unknown"]:
                recognizer.mark_face_seen(face["name"], face["face_id"])

        # 7. Draw results (draw_faces handles conversion internally)
        output_frame = recognizer.draw_faces(gray_frame, faces)
        
        # Convert grayscale to BGR for display
        #output_frame = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)

        # 8. Show instructions
        cv2.putText(output_frame, "R: Register Unknown", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output_frame, "Q: Quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", output_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r') and unknown_faces:
            # Release camera to allow input
            cv2.destroyWindow("Face Recognition")
            name = register_new_face(recognizer, unknown_faces[0])
            if name:
                # Refresh known faces
                recognizer.update_known_faces(db.get_all_faces())
            # Restart camera window
            cv2.namedWindow("Face Recognition")

    cap.release()
    cv2.destroyAllWindows()
    db.close()


if __name__ == "__main__":
    main()