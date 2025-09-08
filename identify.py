import cv2
import argparse
import numpy as np
from collections import deque

# Function to stabilize detections using moving average
def smooth_count(history, new_value, max_size=10):
    history.append(new_value)
    if len(history) > max_size:
        history.popleft()  # Remove oldest value
    return round(sum(history) / len(history))  # Use average to stabilize count

# Face detection function with Non-Maximum Suppression (NMS)
def highlightFace(net, frame, conf_threshold=0.7):
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    confidences = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            confidences.append(float(confidence))

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(faceBoxes, confidences, conf_threshold, 0.4)
    faceBoxes = [faceBoxes[i] for i in indices.flatten()] if len(indices) > 0 else []
    
    return frame, faceBoxes

# Initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--image')
args = parser.parse_args()

# Load models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(20-24)', '(25-40)', '(41-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

video = cv2.VideoCapture(0 if args.image is None else args.image)

if not video.isOpened():
    print("Error: Could not open video source.")
    exit()

padding = 20

# Buffers to stabilize counts over multiple frames
male_count_history = deque(maxlen=10)  # Stores the last 10 detections
female_count_history = deque(maxlen=10)

while True:
    hasFrame, frame = video.read()
    if not hasFrame:
        print("No frame captured, exiting...")
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)

    current_male_count = 0
    current_female_count = 0

    for faceBox in faceBoxes:
        x1, y1, x2, y2 = faceBox
        face = frame[max(0, y1 - padding): min(y2 + padding, frame.shape[0] - 1),
                     max(0, x1 - padding): min(x2 + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        genderConfidence = genderPreds[0].max()  # Get the confidence value of the detected gender
        gender = genderList[genderPreds[0].argmax()] if genderConfidence > 0.7 else "Unknown"

        if gender == 'Male':
            current_male_count += 1
        elif gender == 'Female':
            current_female_count += 1

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        cv2.putText(resultImg, f'{gender}, {age}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Stabilizing the count using a moving average
    stable_male_count = smooth_count(male_count_history, current_male_count)
    stable_female_count = smooth_count(female_count_history, current_female_count)

    # Display the stable counts on the frame
    text = f'Male: {stable_male_count}, Female: {stable_female_count}'
    cv2.putText(resultImg, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    print(f'Male Count: {stable_male_count}, Female Count: {stable_female_count}')

    cv2.imshow("Detecting age and gender", resultImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

video.release()
cv2.destroyAllWindows()
