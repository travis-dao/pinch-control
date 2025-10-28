import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import math

DEFAULT_COLOR = (147,112,219)

def draw_polygon(image, pts, isClosed=True, color=DEFAULT_COLOR, thickness=1, swap=False):
  """
  Draws a polygon outline from an array of points.

  Parameters:
    image (np.ndarray): Input image (BGR format).
    pts (list of [x, y]): List of polygon points.
    isClosed (boolean): close polygon.
    color (3-tuple): line color in (RGB format).
    thickness (int): line thickness.
  """
  # swap elements 3 and 4 to make connected polygon
  if swap:
    temp = pts[2]
    pts[2] = pts[3]
    pts[3] = temp

  # reshape into proper format (idk why but documentation shows this)
  pts = np.array(pts, np.int32)
  pts = pts.reshape((-1, 1, 2))

  # flips color cuz takes in BGR format
  cv2.polylines(image, [pts], isClosed, color[::-1], thickness)


def blur_polygon_area(image, pts, blur_radius=10):
  """
  Blurs a polygon area defined by pts inside a BGR image (OpenCV format).

  Parameters:
    image (np.ndarray): Input image (BGR format).
    pts (list of [x, y]): List of polygon points.
    blur_radius (int): Gaussian blur radius.

  Returns:
    np.ndarray: Image with the polygon area blurred.
  """
  # OpenCV image (BGR) --> PIL (RGB)
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  pil_image = Image.fromarray(image_rgb)

  # [[x, y], ...] --> [(x, y), ...]
  tuple_pts = [tuple(p) for p in pts]

  # mask + blur
  mask = Image.new("L", pil_image.size, 0)
  ImageDraw.Draw(mask).polygon(tuple_pts, fill=255)
  blurred = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))

  # composite blurred region over original
  result = Image.composite(blurred, pil_image, mask)

  # convert back to OpenCV format (BGR)
  result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
  return result_bgr

def draw_line(image, pts, color=DEFAULT_COLOR, thickness=1):
  cv2.line(image, pts[0], pts[1], color[::-1], thickness)

def draw_circle(image, center: cv2.typing.Point, radius, color=(255, 255, 255), thickness=-1):
  cv2.circle(image, center, radius, color[::-1], thickness)

def draw_rect(image, top_left, bottom_right, color=DEFAULT_COLOR, thickness=2):
  cv2.rectangle(image, top_left, bottom_right, color, thickness)

def draw_text(image, text, origin: cv2.typing.Point, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2):
  cv2.putText(image, text, origin, fontFace, fontScale, color[::-1], thickness)

def is_pinching(dist):
  return dist <= 0.05

def get_closest_idx(vertices, pt):
  closest_dist = 99999999999
  dist_list = []
  for vertex in vertices:
    dist = math.dist(pt, vertex)
    dist_list.append(dist)
    if (dist < closest_dist):
      closest_dist = dist
  
  return dist_list.index(closest_dist)


def is_point_inside_circle(center_x, center_y, radius, point_x, point_y):
  """
  Checks if a given point is inside or on the boundary of a circle.

  Args:
      center_x (float): The x-coordinate of the circle's center.
      center_y (float): The y-coordinate of the circle's center.
      radius (float): The radius of the circle.
      point_x (float): The x-coordinate of the point to check.
      point_y (float): The y-coordinate of the point to check.

  Returns:
      bool: True if the point is inside or on the circle, False otherwise.
  """
  # Calculate the squared distance between the point and the circle's center
  # Squaring avoids the need for a square root, which is computationally cheaper
  squared_distance = (point_x - center_x)**2 + (point_y - center_y)**2

  # Compare the squared distance to the squared radius
  return squared_distance <= radius**2