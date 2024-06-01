import cv2
from bruiser_detection import bruiser_detector, rectangle_filler

VIDEO_PATH   = 'videos/20240216_094022.mp4'

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened(): exit('Error')

fps_rate    = cap.get(cv2.CAP_PROP_FPS) # Obtenha a taxa de frames por segundo do vídeo
WAIT_TIME   = int(1000 / fps_rate) # Calculate the wait time to play the video in real-time (1x)
FRAME_CAP   = 6 # Set the seconds between frame captures
interval    = int(fps_rate * FRAME_CAP) # Calculate the interval in frames
frame_index = 0

while cap.isOpened():
  ret, frame = cap.read()

  # cv2.imshow('Video', frame)

  if ret:
    if frame_index % interval == 0:

      frame = bruiser_detector(frame)
      frame_name = f'frames/frame_{int(frame_index/fps_rate):04d}.jpg'
      cv2.imwrite(frame_name, frame)
      print(f'Saving {frame_name}')

    frame_index += 1

    # if cv2.waitKey(WAIT_TIME) & 0xFF == ord('q'):
    #   break
  else:
    break

cap.release()
cv2.destroyAllWindows()