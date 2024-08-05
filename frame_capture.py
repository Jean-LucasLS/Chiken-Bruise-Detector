import cv2

from bruiser_detection import bruiser_detector

VIDEO_PATH   = 'videos/Frango1.mp4'
FRAME_CAP    = 1 # Set the seconds between frame captures

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened(): exit('Error')

fps_rate    = cap.get(cv2.CAP_PROP_FPS) # Get the frame rate per second
interval    = int(fps_rate * FRAME_CAP) # Calculate the interval in frames
frame_index = 0

WAIT_TIME   = int(1000 / fps_rate) # Calculate the wait time to play the video in real-time (1x)

while cap.isOpened():
  ret, frame = cap.read()

  if ret:
    if frame_index % interval == 0:

      frame = bruiser_detector(frame)

      frame_name = f'frames/frame_{int(frame_index/fps_rate):04d}.jpg'
      cv2.imwrite(frame_name, frame)
      print(f'Saving {frame_name}')

    frame_index += 1

    if cv2.waitKey(WAIT_TIME) & 0xFF == ord('q'): # Press 'q' to stop the video
      break

  else:
    break

  cv2.imshow('Video', frame)

cap.release()
cv2.destroyAllWindows()