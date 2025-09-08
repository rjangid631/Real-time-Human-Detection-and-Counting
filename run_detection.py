import cv2
import numpy as np
import json
import time

def highlightFace(net, frame, conf_threshold=0.7):
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
    
    return frame, faceBoxes

def update_counts(male, female):
    """Update detected gender counts in a JSON file."""
    counts = {"male": male, "female": female}
    with open("detection_counts.json", "w") as file:
        json.dump(counts, file)

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

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open video source.")
    exit()

padding = 20

while True:
    hasFrame, frame = video.read()
    if not hasFrame:
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
        gender = genderList[genderPreds[0].argmax()]

        if gender == 'Male':
            current_male_count += 1
        elif gender == 'Female':
            current_female_count += 1

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        cv2.putText(resultImg, f'{gender}, {age}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    text = f'Male: {current_male_count}, Female: {current_female_count}'
    cv2.putText(resultImg, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Store last detected count in a file instead of session
    update_counts(current_male_count, current_female_count)

    cv2.imshow("Gender Detection", resultImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
