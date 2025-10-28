import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import overlay

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
  model_complexity=0,
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      h, w, c = image.shape
      num_hands = len(results.multi_hand_landmarks)

      if num_hands == 2:
        coords = []
        landmark_idxs = [4, 8]

        # for each hand --> get points
        for hand_landmarks in results.multi_hand_landmarks:
          pts = [hand_landmarks.landmark[4], hand_landmarks.landmark[8]] # 4 = thumb tip, 8 = pointer tip
          hand_coords = [[int(p.x * w), int(p.y * h)] for p in pts] # normalized to coordinate points
          coords.extend(hand_coords)

          for idx, landmark in enumerate(hand_landmarks.landmark):
            if idx in landmark_idxs:
              # draw circle
              pos = (int(landmark.x * w), int(landmark.y * h))
              overlay.draw_circle(image, pos, 30)

        # outline
        overlay.draw_polygon(image, coords, color=(255, 192, 203), thickness=5, swap=True)

        # blur
        image = overlay.blur_polygon_area(image, coords)

    # flip the image horizontally = selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()