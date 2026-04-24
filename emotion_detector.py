
"""
Emotion detection module using the FER (Facial Expression Recognition) library.
Classifies emotions from detected face regions.
"""

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np
from fer.fer import FER

logger = logging.getLogger(__name__)

# Supported emotions
EMOTIONS = ["Happy", "Sad", "Angry", "Disgusted", "Surprised", "Fearful", "Neutral"]


class EmotionDetector:
    """Handles emotion detection from face regions."""

    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize the emotion detector.

        Args:
            min_confidence: Minimum confidence threshold for emotion detection.
        """
        self.min_confidence = min_confidence
        self.detector = FER(mtcnn=True)


    def detect_emotion(self, face_image: np.ndarray) -> str:
        """Detect emotion from a face image."""
        try:
            result = self.detector.detect_emotions(face_image)
            if result and result[0]["emotions"]:
                emotions = result[0]["emotions"]
                emotion = max(emotions, key=emotions.get)
                if emotions[emotion] >= self.min_confidence:
                    return emotion.capitalize()
            return "Neutral"
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return "Neutral"