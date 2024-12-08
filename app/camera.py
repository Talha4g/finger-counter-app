import cv2
import mediapipe as mp

class FingerCounter:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
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
