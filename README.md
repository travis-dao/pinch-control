# Pinch Control

**Pinch Control** is a fun computer vision project that plays around with [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) and [OpenCV](https://docs.opencv.org/4.x/) to turn hand-gestures into real-time visuals.

The project showcases **gesture-based control** through pinching and movement tracking.  

It includes two main scripts:

1. `dual_hand_pinch_control.py` – Lets you pinch and drag polygon corners to reshape a region interactively.
2. `hand_outline.py` – Uses both hands to create and control a blurred polygon region.

Both scripts rely on an `overlay.py` helper module for drawing shapes, text, and blurring areas.

Blurring is achieved with the [Pillow (PIL Fork)](https://pillow.readthedocs.io/en/stable/index.html#) library.