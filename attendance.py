import cv2
import pandas as pd
from datetime import datetime

model = cv2.face.LBPHFaceRecognizer_create()
model.read("face_model.yml")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

label_map = {
    0: "Himanshu",
    1: "Rahul"
}

attendance = pd.DataFrame(columns=["Name", "Time"])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        label, confidence = model.predict(face_img)

        if confidence < 80:
            name = label_map[label]
            time = datetime.now().strftime("%H:%M:%S")

            if name not in attendance["Name"].values:
                attendance.loc[len(attendance)] = [name, time]
                attendance.to_csv("attendance.csv", index=False)

            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Smart Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()