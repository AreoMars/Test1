from ultralytics import YOLO
import cv2
from _datetime import datetime

width = 1920
height = 1080
cap = cv2.VideoCapture(r"C:\PythonProjects\Motor_packing\Motor_packing_video\lowwer_unit\ok lower unit.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Load a model
model = YOLO("best242.pt")  # pretrained YOLO11n model

while True:
    success, frame = cap.read()

    if success:

        # Run YOLO inference on the frame
        results = model(frame)
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            class_indices = boxes.cls  # Class indices of the detections
            class_names = [result.names[int(cls)] for cls in class_indices]  # Map indices to names
            print(class_names)
            cls_data = class_names
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        new_annotated_frame = cv2.resize(annotated_frame, (0, 0), fx = 0.7, fy = 0.7)
        # Display the annotated frame
        cv2.imshow("YOLO Inference", new_annotated_frame)
        print(len(cls_data))
        if"BOX" in cls_data:
            print("wati to go next step")


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()