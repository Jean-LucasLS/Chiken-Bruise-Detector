import cv2
import numpy as np

# Define the frame interest area
def cut_frame(frame):
  height, width = frame.shape[:2]
  fr_x_start    = int(width * 0.25)
  fr_x_end      = int(width * 0.75)
  fr_y_start    = int(height * 0.20)
  fr_y_end      = int(height * 0.80)

  frame = frame[fr_y_start:fr_y_end, fr_x_start:fr_x_end]
  return frame

def rectangle_filler(frame, contours):
  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
  return frame

def bruiser_detector(frame):
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert the frame to HSV color space

  frame = cut_frame(frame)

  # Define range for the color that could represent a bruise
  lower_color1 = np.array([0, 50, 50])
  upper_color1 = np.array([10, 255, 255])
  lower_color2 = np.array([160, 50, 50])
  upper_color2 = np.array([180, 255, 255])
  
  # Create masks for the lower and upper color ranges
  mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
  mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
  mask  = cv2.bitwise_or(mask1, mask2)

  # Dilation to increase the areas size detection
  kernel = np.ones((50, 50), np.uint8)
  mask   = cv2.dilate(mask, kernel, iterations=1)
  
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours in the mask

  frame = rectangle_filler(frame, contours)
  return frame