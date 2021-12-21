# Importing the dependencies
import cv2
import numpy as np

# Defining variables to store coordinates where the second image has to be placed
positions = []
positions2 = []
count = 0


# Mouse callback function
def draw_circle(event, x, y, flags, param):
    global positions, count
    # If event is Left Button Click then store the coordinate in the lists, positions and positions2
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(car, (x, y), 2, (255, 0, 0), -1)
        positions.append([x, y])
        if (count != 3):
            positions2.append([x, y])
        elif (count == 3):
            positions2.insert(2, [x, y])
        count += 1


# car = cv2.imread('img/cartype167/car1.jpg')
car = cv2.imread('img/cartype167/car3.jpg')
plate = cv2.imread('license1.jpg')

# Defing a window named 'image'
cv2.namedWindow('image')

cv2.setMouseCallback('image', draw_circle)

while (True):
    cv2.imshow('image', car)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

print(positions)
print(positions2)

# Getting the coordinates of corners from the first image
height, width = car.shape[:2]
h1, w1 = plate.shape[:2]

# pts1 = np.float32([[0, 0], [w1, 0], [0, h1], [w1, h1]])
pts1 = np.float32([[0, 0], [0, h1], [w1, 0], [w1, h1]])

pts2 = np.float32(positions)

h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
#print homography matix
# print(h)

height, width, channels = car.shape
im1Reg = cv2.warpPerspective(plate, h, (width, height))

mask2 = np.zeros(car.shape, dtype=np.uint8)

roi_corners2 = np.int32(positions2)

channel_count2 = car.shape[2]
ignore_mask_color2 = (255,) * channel_count2

cv2.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)

mask2 = cv2.bitwise_not(mask2)
masked_image2 = cv2.bitwise_and(car, mask2)
# cv2.imwrite('./img/test_car/masked.png', masked_image2)

# Using Bitwise or to merge the two images
final = cv2.bitwise_or(im1Reg, masked_image2)
cv2.imwrite('./img/test_car/addplate.png', final)
