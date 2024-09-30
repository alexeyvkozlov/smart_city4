#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd с:\my_campy\smart_city\check_utilities
# cd d:\my_campy\smart_city\check_utilities
#~~~~~~~~~~~~~~~~~~~~~~~~
# python read_video_opencv.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('~'*70)

# Create a VideoCapture object and read from input file
# To capture a video, you need to create a VideoCapture object
cap = cv2.VideoCapture('vtest.avi')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  #cap.read() returns a bool (True/False). If the frame is read correctly, it will be True. 
  #So you can check for the end of the video by checking this returned value.
  ret, frame = cap.read()    
  if ret == True:
    # Display the resulting frame
    cv2.imshow('Frame',frame)
    print(f'frame.shape: {frame.shape}')

    # Press Q on keyboard to  exit
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    if 27 == cv2.waitKey(25):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('='*70)
print('finish program')
