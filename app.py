from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64
import os
from waitress import serve

app = Flask(__name__)
CORS(app)

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
                    
                    # Draw count for each hand
                    label = f'{handedness.classification[0].label}: {finger_count}'
                    cv2.putText(image, label, (text_x, 50 + idx * 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                    cv2.putText(image, label, (text_x, 50 + idx * 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Draw total count
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        if not request.json or 'image' not in request.json:
            return jsonify({'error': 'No image data received'}), 400

        # Get the image data from the request
        image_data = request.json['image']
        if not image_data:
            return jsonify({'error': 'Empty image data'}), 400

        try:
            image_data = image_data.split(',')[1]
        except IndexError:
            return jsonify({'error': 'Invalid image data format'}), 400

        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({'error': f'Base64 decode error: {str(e)}'}), 400

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Could not decode image'}), 400

        processed_frame = counter.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'image': f'data:image/jpeg;base64,{processed_image}'})
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return {'status': 'healthy'}, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    
    if os.environ.get('ENVIRONMENT') == 'production':
        serve(app, host='0.0.0.0', port=port)
    else:
        app.run(host='0.0.0.0', port=port, debug=False)