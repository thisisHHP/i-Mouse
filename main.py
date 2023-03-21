import cv2
import mediapipe as mp
import pyautogui
screen_w, screen_h = pyautogui.size()
facem_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks = True)
cam = cv2.VideoCapture(0)
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = facem_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x*frame_w)
            y = int(landmark.y*frame_h)
            cv2.circle(frame, (x,y), 3, (0,255,0))
            if id ==1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                print(x,y)
                pyautogui.moveTo(screen_x, screen_y)

    print(landmark_points)
    cv2.imshow('i-Mouse', frame)
    cv2.waitKey(1)