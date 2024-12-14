from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time
import os
from waitress import serve

app = Flask(__name__)

# Configure for production
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['DEBUG'] = False

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
            
        # Other fingers - same for both hands
        for id in range(1, 5):
            if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers.count(1)

    def process_frame(self, frame):
        try:
            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            total_fingers = 0
            
            if results.multi_hand_landmarks:
                for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                        self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # Count fingers
                    finger_count = self.count_fingers(hand_landmarks, handedness)
                    total_fingers += finger_count
                    
                    # Position text based on hand
                    text_x = 50 if handedness.classification[0].label == 'Left' else image.shape[1] - 250
                    
                    # Draw count for each hand with outline
                    label = f'{handedness.classification[0].label}: {finger_count}'
                    cv2.putText(image, label, (text_x, 50 + idx * 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                    cv2.putText(image, label, (text_x, 50 + idx * 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Draw total count with outline
                total_text = f'Total Fingers: {total_fingers}'
                cv2.putText(image, total_text, (image.shape[1]//2 - 100, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                cv2.putText(image, total_text, (image.shape[1]//2 - 100, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            return image
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame

counter = FingerCounter()

def get_camera():
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
    except Exception as e:
        print(f"Camera error: {e}")
    return None

camera = None

def generate_frames():
    global camera
    
    while True:
        try:
            if camera is None:
                camera = get_camera()
                if camera is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Camera not available", (200, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    time.sleep(1)
                    continue

            success, frame = camera.read()
            if not success:
                raise Exception("Failed to read camera frame")

            frame = cv2.resize(frame, (640, 480))
            processed_frame = counter.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(0.1)
            continue

    if camera is not None:
        camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    return {'status': 'healthy'}, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    
    if os.environ.get('ENVIRONMENT') == 'production':
        serve(app, host='0.0.0.0', port=port)
    else:
        app.run(host='0.0.0.0', port=port, debug=False)