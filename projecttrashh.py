import cv2
import numpy as np
import tensorflow as tf
from collections import deque

model = tf.keras.models.load_model('model2.h5')
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def preprocess_frame(frame):
    frame = cv2.GaussianBlur(frame, (5, 5), 0) 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    return np.expand_dims(normalized_frame, axis=0).astype(np.float32)

def get_weighted_prediction(buffer, confidences):
    weighted_sum = np.zeros(len(class_labels))
    for idx, conf in zip(buffer, confidences):
        weighted_sum[idx] += conf
    return np.argmax(weighted_sum)

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

prediction_buffer = deque(maxlen=10) 
confidence_buffer = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    fgmask = fgbg.apply(frame)
    fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)
    _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 2000:
            x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = max(0, x), max(0, y), min(frame.shape[1]-x, w), min(frame.shape[0]-y, h)
            
            detected_object = frame[y:y+h, x:x+w]
            if detected_object.size == 0:
                continue

            input_data = preprocess_frame(detected_object)
            predictions = model.predict(input_data)
            predicted_class_idx = np.argmax(predictions)
            classification_confidence = np.max(predictions) * 100

            prediction_buffer.append(predicted_class_idx)
            confidence_buffer.append(classification_confidence)

            smoothed_class_idx = get_weighted_prediction(prediction_buffer, confidence_buffer)
            smoothed_class = class_labels[smoothed_class_idx]
            avg_confidence = np.mean(confidence_buffer)

            label = f"{smoothed_class} ({avg_confidence:.2f}%)"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Waste Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
