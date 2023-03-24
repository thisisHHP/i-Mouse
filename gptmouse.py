import cv2
import pyautogui

# Set the camera resolution and framerate
width, height = 640, 480
fps = 30

# Create a VideoCapture object
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

# Define the cascade classifier to detect the eye
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Set the size of the pupil and the sensitivity of the mouse movement
pupil_size = 10
sensitivity = 20

while True:
    # Capture a frame from the cameras
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray)

    # Loop through each eye
    for (x, y, w, h) in eyes:
        # Get the region of interest (ROI) for the eye
        roi_gray = gray[y:y+h, x:x+w]

        # Detect the pupil in the ROI
        circles = cv2.HoughCircles(roi_gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=0, maxRadius=pupil_size)

        # If a pupil was found, move the mouse cursor
        if circles is not None:
            # Get the position of the pupil
            (pupil_x, pupil_y, pupil_r) = circles[0][0]

            # Calculate the position of the mouse cursor based on the pupil position
            mouse_x = int((x + pupil_x - (pupil_size/2)) * sensitivity)
            mouse_y = int((y + pupil_y - (pupil_size/2)) * sensitivity)

            # Move the mouse cursor
            pyautogui.moveTo(mouse_x, mouse_y)

    # Display the frame
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()