
"""
Face recognition module using dlib-based face_recognition library.
Handles face detection, encoding, and recognition with confidence thresholds.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np

from database import get_database

logger = logging.getLogger(__name__)

# Recognition settings
KNOWN_FACE_TOLERANCE = 0.5  # Lower = stricter match (0.4-0.6 recommended)
FRAME_SKIP = 3  # Process every Nth frame
ENCODING_MODEL = "hog"  # 'hog' is faster, 'cnn' is more accurate but slower

class FaceRecognizer:
    """Handles face detection, encoding, and recognition."""

    def __init__(self, known_faces: Optional[List[Dict]] = None):
        """
        Initialize the face recognizer.
    
        Args:
            known_faces: List of dicts with 'name' and 'face_encoding' keys.
        """
        self.known_faces = known_faces or []
        self.frame_count = 0
        self._last_results: List[Dict] = []
        self._next_face_id = 0

        logger.info(
            f"FaceRecognizer initialized with {len(self.known_faces)} known faces"
        )

    def register_new_face(self, name: str, face_encoding) -> bool:
        """Register a new face in the database."""
        db = get_database()
        return db.register_face(name, face_encoding)

    def mark_face_seen(self, name: str, face_id: str) -> bool:
        """Update last_seen timestamp for a face."""
        db = get_database()
        return db.update_last_seen(name, face_id)

    def update_known_faces(self, known_faces: List[Dict]) -> None:
        """Update the known faces database."""
        self.known_faces = known_faces
        logger.info(f"Updated known faces database: {len(known_faces)} faces")

    def _get_next_face_id(self) -> str:
        """Generate a new unique face ID."""
        face_id = f"face_{self._next_face_id}"
        self._next_face_id += 1
        return face_id

    def _find_match(
        self, face_encoding
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Find if a face encoding matches any known face.

        Args:
            face_encoding: 128-dimensional face encoding array.

        Returns:
            Tuple of (face_id, name, confidence) - None values if no match.
        """
        if not self.known_faces:
            return None, None, 0.0

        known_encodings = []
        known_names = []
        known_ids = []

        for face in self.known_faces:
            known_encodings.append(face["face_encoding"])
            known_names.append(face["name"])
            known_ids.append(face.get("face_id", "unknown"))

        distances = face_recognition.face_distance(known_encodings, face_encoding)

        if len(distances) == 0:
            return None, None, 0.0

        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]

        # Convert distance to confidence (inverse relationship)
        confidence = max(0.0, 1.0 - best_distance)

        if best_distance <= KNOWN_FACE_TOLERANCE:
            return known_ids[best_idx], known_names[best_idx], confidence

        return None, None, 0.0

    def encode_face(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face encoding from an image.

        Args:
            face_image: Cropped face image array (BGR or grayscale format).

        Returns:
            128-dimensional encoding array, or None if encoding fails.
        """
        try:
            # Handle grayscale input - convert to RGB
            if len(face_image.shape) == 2:
                # Grayscale - convert to RGB by stacking
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            else:
                # BGR - convert to RGB
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            encodings = face_recognition.face_encodings(
                rgb_image, model=ENCODING_MODEL
            )
            if encodings:
                return encodings[0]
            return None
        except Exception as e:
            logger.error(f"Failed to encode face: {e}")
            return None

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a video frame to detect and recognize faces.
        Uses frame skipping for performance optimization.

        Args:
            frame: Video frame in BGR or grayscale format.

        Returns:
            List of detected face information dicts.
        """
        self.frame_count += 1

        # Skip frames to maintain real-time performance
        if self.frame_count % FRAME_SKIP != 0:
            return self._last_results

        # Handle grayscale input - convert to RGB
        if len(frame.shape) == 2:
            # Grayscale - convert to RGB by stacking channels
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            # BGR - convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            face_locations = face_recognition.face_locations(
                rgb_frame, model=ENCODING_MODEL
            )
            face_encodings = face_recognition.face_encodings(
                rgb_frame, face_locations, model=ENCODING_MODEL
            )
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return self._last_results

        results = []

        for idx, (face_location, face_encoding) in enumerate(
            zip(face_locations, face_encodings)
        ):
            top, right, bottom, left = face_location

            face_id, name, confidence = self._find_match(face_encoding)

            # Assign face ID if new unknown face
            if face_id is None:
                face_id = self._get_next_face_id()

            result = {
                "face_id": face_id,
                "name": name,
                "location": face_location,
                "confidence": confidence,
                "is_unknown": name is None,
                "encoding": face_encoding,
                "bounding_box": (top, right, bottom, left),
            }
            results.append(result)

        self._last_results = results
        return results

    def draw_faces(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.

        Args:
            frame: Video frame in BGR or grayscale format.
            faces: List of face information dicts.

        Returns:
            Frame with drawn annotations.
        """
        # Create a copy to avoid modifying original
        output = frame.copy()
        
        # Only convert if grayscale (1 channel)
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

        for face in faces:
            top, right, bottom, left = face["bounding_box"]
            name = face["name"] or "Unknown"
            confidence = face["confidence"]
            emotion = face.get("emotion", "Neutral")

            # Draw bounding box
            color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
            cv2.rectangle(output, (left, top), (right, bottom), color, 2)

            # Draw name label background
            label = f"{name} ({confidence:.0%})" if name != "Unknown" else "Unknown"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                output,
                (left, top - label_size[1] - 10),
                (left + label_size[0], top),
                color,
                cv2.FILLED,
            )

            # Draw name text
            cv2.putText(
                output,
                label,
                (left, top - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

            # Draw emotion label
            emotion_label = f"Emotion: {emotion}"
            cv2.putText(
                output,
                emotion_label,
                (left, bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        return output