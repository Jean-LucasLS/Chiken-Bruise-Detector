import cv2
import numpy as np

def rectangle_filler(frame, contours):
  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
  return frame

def bruiser_detector(frame):
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert the frame to HSV color space

  # Define range for the color that could represent a bruise (e.g., shades of blue/purple)
  lower_red = np.array([0, 50, 50])
  upper_red = np.array([10, 255, 255])

  # Create a mask with the specified color range
  mask = cv2.inRange(hsv, lower_red, upper_red)
  
  kernel = np.ones((100, 100), np.uint8)
  mask = cv2.dilate(mask, kernel, iterations=1)

  # Find contours in the mask
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  frame = rectangle_filler(frame, contours)
  return frame