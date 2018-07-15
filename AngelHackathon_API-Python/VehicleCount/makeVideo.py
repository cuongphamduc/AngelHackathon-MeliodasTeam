import cv2
import glob

images = glob.glob('./out/*.png')
a = []
for image in images:
    a.append(image)
img = cv2.imread(a[0])
height, width, channels = img.shape
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (width, height))
for i in a :
    img = cv2.imread(i)
    out.write(img)
out.release()

