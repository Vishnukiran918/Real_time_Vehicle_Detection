import cv2

# Load the Haar cascades for pedestrian and car detection
pedestrian_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Open the video capture device
cap = cv2.VideoCapture('sample.mp4')

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect pedestrians and cars in the frame
    pedestrians = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Print a message based on the detected objects
    if len(pedestrians) > 0:
        print("Brake! Pedestrian detected!")
    elif len(cars) > 0:
        print("Accelerate! Car detected!")
    else:
        print("Steer! No obstacles detected.")

    # Draw rectangles and labels around the detected objects in the frame
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Pedestrian', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'Car', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show the frame with the detected objects
    cv2.imshow('frame', frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
