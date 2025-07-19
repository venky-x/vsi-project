import cv2
import mediapipe as mp
import math
import face_recognition
import numpy as np

# Initialize MediaPipe Hands and FaceMesh modules
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.5)

# Load known face encodings and names
known_face_encodings = []
known_face_names = []

# Load known face images and get their encodings
face_image_1 = face_recognition.load_image_file("/Users/shadow/Desktop/Rajagopal Vengates_VSI_CW2/Known_person.jpg")
face_image_2 = face_recognition.load_image_file("/Users/shadow/Desktop/Rajagopal Vengates_VSI_CW2/Known_person2.jpg")
face_image_3 = face_recognition.load_image_file("/Users/shadow/Desktop/Rajagopal Vengates_VSI_CW2/Known_person3.jpg")

# Generate encodings for the loaded faces
known_face_encodings.append(face_recognition.face_encodings(face_image_1)[0])
known_face_names.append("venky")

known_face_encodings.append(face_recognition.face_encodings(face_image_2)[0])
known_face_names.append("rajni")

known_face_encodings.append(face_recognition.face_encodings(face_image_3)[0])
known_face_names.append("vijay")
# Initialize OpenCV to capture webcam feed
cap = cv2.VideoCapture(0)

# Function to calculate the distance between two points
def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Function to classify the emotion based on facial landmarks
def classify_emotion(landmarks):
    # Get key facial landmarks
    mouth_top = landmarks[13]  # Top middle of the mouth
    mouth_bottom = landmarks[14]  # Bottom middle of the mouth
    mouth_left = landmarks[61]  # Left corner of the mouth
    mouth_right = landmarks[291]  # Right corner of the mouth
    left_eye = landmarks[33]  # Left eye center
    right_eye = landmarks[133]  # Right eye center
    left_eyebrow = landmarks[46]  # Left eyebrow
    right_eyebrow = landmarks[276]  # Right eyebrow

    # Calculate distances between key landmarks
    mouth_width = euclidean_distance((mouth_left.x, mouth_left.y), (mouth_right.x, mouth_right.y))
    mouth_height = euclidean_distance((mouth_top.x, mouth_top.y), (mouth_bottom.x, mouth_bottom.y))
    eye_distance = euclidean_distance((left_eye.x, left_eye.y), (right_eye.x, right_eye.y))
    eyebrow_distance = euclidean_distance((left_eyebrow.x, left_eyebrow.y), (right_eyebrow.x, right_eyebrow.y))

    # Classify based on facial landmark ratios and distances
    if mouth_height > 0.2 * eye_distance and mouth_width > 0.6 * eye_distance:
        return "Happy"
    elif mouth_height > 0.3 * eye_distance and eyebrow_distance > 0.3 * eye_distance:
        return "Sad"
    elif eyebrow_distance > 0.3 * eye_distance and mouth_width < 0.5 * eye_distance:
        return "Angry"
    else:
        return "Neutral"

# Function to classify hand gestures
def classify_hand_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Check for Thumbs Up Gesture
    if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y and thumb_tip.y < ring_tip.y and thumb_tip.y < pinky_tip.y:
        # Ensure the other fingers are curled down
        if index_tip.y > landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and \
                middle_tip.y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and \
                ring_tip.y > landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y and \
                pinky_tip.y > landmarks[mp_hands.HandLandmark.PINKY_PIP].y:
            return "Thumbs Up"
    elif abs(thumb_tip.y - index_tip.y) < 0.02 and abs(index_tip.y - middle_tip.y) < 0.02:
        return "Fist"
    elif abs(thumb_tip.y - pinky_tip.y) > 0.05:
        return "hello"
    else:
        return "Unknown Gesture"

# Function to count fingers
def count_fingers(landmarks):
    finger_count = 0
    if landmarks[4].x < landmarks[3].x:  # Thumb
        finger_count += 1
    if landmarks[8].y < landmarks[6].y:  # Index
        finger_count += 1
    if landmarks[12].y < landmarks[10].y:  # Middle
        finger_count += 1
    if landmarks[16].y < landmarks[14].y:  # Ring
        finger_count += 1
    if landmarks[20].y < landmarks[18].y:  # Pinky
        finger_count += 1
    return finger_count

# Set webcam resolution to reduce processing load
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_counter = 0
skip_frames = 2  # Skip every 2nd frame to reduce processing load

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Skip frames to reduce processing load
    if frame_counter % skip_frames == 0:
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face Recognition Processing
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Process only when faces are found
        if face_locations:
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Process emotion and hand gesture only every nth frame
        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        # Emotion detection (FaceMesh)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                emotion = classify_emotion(face_landmarks.landmark)
                cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hand gesture detection and finger counting
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                gesture = classify_hand_gesture(hand_landmarks.landmark)
                finger_count = count_fingers(hand_landmarks.landmark)
                cv2.putText(frame, f"Gesture: {gesture}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Fingers: {finger_count}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the processed frame
        cv2.imshow('Face Recognition, Emotion and Gesture Detection', frame)

    frame_counter += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
