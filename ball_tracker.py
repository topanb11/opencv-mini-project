import cv2 as cv

# upload video
video_capture = cv.VideoCapture('assets/basketball-swish.mp4')

# add background subtractor
backsub = cv.createBackgroundSubtractorMOG2()

while video_capture.isOpened():
	# read each frame
  ret, frame = video_capture.read()
  if not ret:
    break
  
  # convert video to gray-scale and add mask to track motion
  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  fgMask = backsub.apply(gray)
  
  # find contours of moving object
  contours, _ = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  
  # process each of the contours
  for contour in contours:
    if cv.contourArea(contour) < 1000 or cv.contourArea(contour) > 5000:
      continue
    
    # get coordinates of contour for rectangle
    (x, y, w, h) = cv.boundingRect(contour)
    square_size = max(w,h)
    
    x_centered = x + w // 2 - square_size // 2
    y_centered = y + h // 2 - square_size // 2
    
    top_left = (x_centered, y_centered)
    bottom_right = (x_centered + square_size, y_centered + square_size)
    
    # draw rectangle around moving object
    cv.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

  cv.imshow('Square ball tracker', frame)
  cv.imshow('Square ball tracker FG Mask', fgMask)
	
  if cv.waitKey(30) == ord('q'):
    break

video_capture.release()
cv.destroyAllWindows()