import cv2
import mediapipe as mp
import math
import overlay

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# --- initial text position ---
vertices = [
  [300, 300],
  [1500, 300],
  [1500, 900],
  [300, 900]
]

# --- drag states for both hands ---
dragging = {
  "Left": False,
  "Right": False
}

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
  model_complexity=0,
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5
) as hands:

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

    h, w, _ = image.shape


    drag_offset = {
      "Left": (0, 0), 
      "Right": (0, 0)
    }

    # store distances + hand labels
    hand_dists = {}
    hand_coords = {}
    hand_midpoints = {}

    if results.multi_hand_landmarks:
      handedness_list = results.multi_handedness

      for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
        label = handedness_list[i].classification[0].label  # "Left" or "Right"
        label = 'Right' if label == 'Left' else 'Left'

        # get coordinates
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]

        thumb_xy = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        index_xy = (int(index_tip.x * w), int(index_tip.y * h))

        overlay.draw_circle(image, thumb_xy, 30, thickness=5)
        overlay.draw_circle(image, index_xy, 30, thickness=5)

        # compute 2D distance (for pinch detection)
        dist = math.dist(
          [thumb_tip.x, thumb_tip.y],
          [index_tip.x, index_tip.y]
        )

        hand_dists[label] = dist
        hand_coords[label] = (thumb_xy, index_xy)

        # compute midpoint for drag
        mid_x = int((thumb_xy[0] + index_xy[0]) / 2)
        mid_y = int((thumb_xy[1] + index_xy[1]) / 2)
        hand_midpoints[label] = (w - mid_x, mid_y)  # flip horizontally for selfie view

    flipped_image = cv2.flip(image, 1)
    
    overlay.draw_text(flipped_image, f'Hands: {list(hand_dists.keys())}', (50, 50))

    padding = 20
    radius = 30
    
    overlay.draw_polygon(flipped_image, vertices, thickness=5)
    flipped_image = overlay.blur_polygon_area(flipped_image, vertices)

    # --- handle each hand independently ---
    for label in hand_dists.keys():
      is_pinching = overlay.is_pinching(hand_dists[label])
      # overlay.draw_text(flipped_image, f'Dist: {hand_dists[label]:.4f}', (50, 100 if label == 'Right' else 150))
      pinch_x, pinch_y = hand_midpoints[label]

      # know which vertex to modify
      vertex_idx = overlay.get_closest_idx(vertices, [pinch_x, pinch_y])
      vertex = vertices[vertex_idx]
      x = vertex[0]
      y = vertex[1]

      # debug - draw line to closet vertex
      # overlay.draw_line(flipped_image, [vertex, [pinch_x, pinch_y]], (0, 255, 0), thickness=5)

      # check if touching vertex
      inside_text = overlay.is_point_inside_circle(pinch_x, pinch_y, radius + padding, x, y)

      if is_pinching:
        if inside_text and not dragging[label]:
          dragging[label] = True
          drag_offset[label] = (pinch_x - x, pinch_y - y)
        elif dragging[label]:
          x = pinch_x - drag_offset[label][0]
          y = pinch_y - drag_offset[label][1]
        
        # update vertices with new coords
        vertices[vertex_idx] = [x, y]
      else:
        dragging[label] = False

    # overlay.draw_text(flipped_image, f'Circle: ({x}, {y})', (w - 400, h - 100))

    # draw mask
    for idx, vertex in enumerate(vertices):
      overlay.draw_circle(flipped_image, (vertex[0], vertex[1]), radius, color=overlay.DEFAULT_COLOR, thickness=5)
      # overlay.draw_text(flipped_image, f'Vertex {idx + 1}: ({vertex[0]}, {vertex[1]})', (vertex[0], vertex[1]))

    cv2.imshow('Handtracking Testing', flipped_image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()