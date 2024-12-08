import cv2
import mediapipe as mp
import numpy as np
import os

class FingerCounter:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.camera = None
        self.is_camera_available = False
        self.initialize_camera()

    def initialize_camera(self):
        try:
            # Try to initialize camera
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise ValueError("Could not open camera")
            self.is_camera_available = True
        except Exception as e:
            print(f"Camera initialization failed: {str(e)}")
            self.is_camera_available = False
            
    def get_frame(self):
        if not self.is_camera_available:
            # Return a blank frame with error message if camera is not available
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "Camera not available",
                (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            return frame
            
        ret, frame = self.camera.read()
        if not ret:
            return self.get_frame()  # Recursively try again or return blank frame
        return frame

    def count_fingers(self, hand_landmarks, handedness):
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []
        
        # Thumb - different logic for left and right hands
        if handedness.classification[0].label == 'Right':
            if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:  # Left hand
            if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
            
        # Other fingers
        for id in range(1, 5):
            if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers.count(1)

    def process_frame(self, frame):
        # If we're getting a None frame, return a blank one
        if frame is None:
            return self.get_frame()
            
        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        left_count = 0
        right_count = 0
        total_count = 0
        
        if results.multi_hand_landmarks:
            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                self.mp_draw.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                finger_count = self.count_fingers(hand_landmarks, handedness)
                
                if handedness.classification[0].label == 'Left':
                    left_count = finger_count
                else:
                    right_count = finger_count
                
                text_x = 50 if handedness.classification[0].label == 'Left' else 400
                cv2.putText(
                    image, 
                    f'{handedness.classification[0].label}: {finger_count}', 
                    (text_x, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 0, 0), 
                    2
                )
            
            total_count = left_count + right_count
            cv2.putText(
                image, 
                f'Total: {total_count}', 
                (225, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 0, 0), 
                2
            )
        
        return image
