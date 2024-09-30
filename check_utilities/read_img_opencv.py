#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd с:\my_campy\smart_city\check_utilities
# cd d:\my_campy\smart_city\check_utilities
#~~~~~~~~~~~~~~~~~~~~~~~~
# python read_img_opencv.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('~'*70)
print(f'opencv: {cv2.__version__}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('~'*70)
img = cv2.imread('dog.jpg')

print(f'[INFO] img shape: {img.shape}') 

# show the output image
cv2.imshow("Output",img)
# allows users to display a window for given milliseconds.
# 0 means infinity. 2000 means 2 seconds
cv2.waitKey(0)

#res_img_write = cv2.imwrite('dog2.jpg', img)
#print(f'res_img_write: {res_img_write}') 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('='*70)
print('[INFO] -> program completed!')