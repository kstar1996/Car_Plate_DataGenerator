# Importing the dependencies
import math
import random
import cv2
import numpy as np
import os


def change_light(image, coef):
    # Conversion to HLS
    image_HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_HLS[:, :, 2] = image_HLS[:, :, 2] * coef
    # Conversion to RGB
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HSV2BGR)
    return image_RGB


# make image darker
def darken(image, darkness_coef=-1):
    if darkness_coef == -1:
        darkness_coef_t = 1 - random.uniform(0, 0.8)
    else:
        darkness_coef_t = 1 - darkness_coef
    # Change the light in the image according to the darkness coef
    image_RGB = change_light(image, darkness_coef_t)
    return image_RGB


def flare_source(image, point, radius, src_color):
    overlay = image.copy()
    output = image.copy()
    num_times = radius // 10
    alpha = np.linspace(0.0, 1, num=num_times)
    rad = np.linspace(1, radius, num=num_times)
    for i in range(num_times):
        cv2.circle(overlay, point, int(rad[i]), src_color, -1)
        alp = alpha[num_times - i - 1] * alpha[num_times - i - 1] * alpha[num_times - i - 1]
        cv2.addWeighted(overlay, alp, output, 1 - alp, 0, output)
    return output


def add_sun_flare_line(flare_center, angle, imshape):
    x = []
    y = []
    for rand_x in range(0, imshape[1], 10):
        rand_y = math.tan(angle) * (rand_x - flare_center[0]) + flare_center[1]
        x.append(rand_x)
        y.append(2 * flare_center[1] - rand_y)
    return x, y


def add_sun_process(image, no_of_flare_circles, flare_center, src_radius, x, y, src_color):
    overlay = image.copy()
    output = image.copy()
    imshape = image.shape
    for i in range(no_of_flare_circles):
        alpha = random.uniform(0.05, 0.2)
        r = random.randint(0, len(x) - 1)
        rad = random.randint(1, imshape[0] // 100 - 2)
        cv2.circle(overlay, (int(x[r]), int(y[r])), rad * rad * rad, (
            random.randint(max(src_color[0] - 50, 0), src_color[0]),
            random.randint(max(src_color[1] - 50, 0), src_color[1]),
            random.randint(max(src_color[2] - 50, 0), src_color[2])), -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    output = flare_source(output, (int(flare_center[0]), int(flare_center[1])), src_radius, src_color)
    return output


def sun_flare(image, flare_center=-1, no_of_flare_circles=8, src_radius=400, src_color=(255, 255, 255)):
    angle = -1
    angle = angle % (2 * math.pi)

    if type(image) is list:
        image_RGB = []
        image_list = image
        image_shape = image_list[0].shape
        for img in image_list:
            angle_t = random.uniform(0, 2 * math.pi)
            if angle_t == math.pi / 2:
                angle_t = 0
            if flare_center == -1:
                flare_center_t = (random.randint(0, image_shape[1]), random.randint(0, image_shape[0] // 2))
            else:
                flare_center_t = flare_center
            x, y = add_sun_flare_line(flare_center_t, angle_t, image_shape)
            output = add_sun_process(img, no_of_flare_circles, flare_center_t, src_radius, x, y, src_color)
            image_RGB.append(output)
    else:
        image_shape = image.shape
        if angle == -1:
            angle_t = random.uniform(0, 2 * math.pi)
            if angle_t == math.pi / 2:
                angle_t = 0
        else:
            angle_t = angle
        if flare_center == -1:
            flare_center_t = (random.randint(0, image_shape[1]), random.randint(0, image_shape[0] // 2))
        else:
            flare_center_t = flare_center
        x, y = add_sun_flare_line(flare_center_t, angle_t, image_shape)
        output = add_sun_process(image, no_of_flare_circles, flare_center_t, src_radius, x, y, src_color)
        image_RGB = output
    return image_RGB


def generate_random_circles(imshape, slant, drop_length):
    drops = []
    size = int(imshape[0] * imshape[1] / 300)
    for i in range(size):  # If You want heavy rain, try increasing this
        # if slant<0:
        # 	x= np.random.randint(slant,imshape[1])
        # else:
        x = np.random.randint(0, imshape[1] - slant)
        y = np.random.randint(0, imshape[0] - drop_length)
        drops.append((x, y))
    return drops


# Add rain
def generate_random_lines(imshape, slant, drop_length):
    drops = []
    size = int(imshape[0] * imshape[1] / 300)
    for i in range(size):  # If You want heavy rain, try increasing this
        # if slant<0:
        # 	x= np.random.randint(slant,imshape[1])
        # else:
        x = np.random.randint(0, imshape[1] - slant)
        y = np.random.randint(0, imshape[0] - drop_length)
        drops.append((x, y))
    return drops


def add_rain(image):
    image = image.astype(np.float32)
    imshape = image.shape
    print(imshape)
    slant_extreme = 10
    slant = np.random.randint(-slant_extreme, slant_extreme)
    drop_length = 20
    drop_width = 1
    drop_color = (200, 200, 200)  # a shade of gray
    rain_drops = generate_random_lines(imshape, slant, drop_length)
    # print('rain_drops',len(rain_drops))
    for rain_drop in rain_drops:
        cv2.line(image, (rain_drop[0], rain_drop[1]), (rain_drop[0] + slant, rain_drop[1] + drop_length), drop_color,
                 drop_width)
    image = cv2.blur(image, (3, 3))  # increase number if you want more blur
    brightness_coefficient = 0.9  # rainy days are usually shady
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)  # Conversion to HLS
    image_HLS[:, :, 1] = image_HLS[:, :, 1] * brightness_coefficient  # scale pixel values down for channel 1(Lightness)
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)  # Conversion to RGB
    return image_RGB


def add_snow(image):
    image = image.astype(np.float32)
    imshape = image.shape
    print(imshape)
    slant_extreme = 10
    slant = np.random.randint(-slant_extreme, slant_extreme)
    drop_length = 2
    drop_width = 2
    drop_color = (200, 200, 200)  # a shade of gray
    rain_drops = generate_random_circles(imshape, slant, drop_length)
    # print('rain_drops',len(rain_drops))
    for rain_drop in rain_drops:
        cv2.line(image, (rain_drop[0], rain_drop[1]), (rain_drop[0] + slant, rain_drop[1] + drop_length), drop_color,
                 drop_width)
    image = cv2.blur(image, (2, 2))  # increase number if you want more blur
    brightness_coefficient = 0.9  # rainy days are usually shady, increase for more brightness
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)  # Conversion to HLS
    image_HLS[:, :, 1] = image_HLS[:, :, 1] * brightness_coefficient  # scale pixel values down for channel 1(Lightness)
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)  # Conversion to RGB
    return image_RGB


#  shadow
def generate_shadow_coordinates(imshape, no_of_shadows=1):
    vertices_list = []
    for index in range(no_of_shadows):
        vertex = []
        for dimensions in range(np.random.randint(3, 15)):  # Dimensionality of the shadow polygon
            vertex.append((imshape[1] * np.random.uniform(), imshape[0] // 3 + imshape[0] * np.random.uniform()))
        vertices = np.array([vertex], dtype=np.int32)  # single shadow vertices
        vertices_list.append(vertices)
    return vertices_list  # List of shadow vertices


def add_shadow(image, no_of_shadows=3):
    image = image.astype(np.float32)
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # Conversion to HLS
    mask = np.zeros_like(image)
    imshape = image.shape
    vertices_list = generate_shadow_coordinates(imshape, no_of_shadows)
    # 3 getting list of shadow vertices
    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices,
                     255)  # adding all shadow polygons on empty mask, single 255 denotes only red channel
    image_HLS[:, :, 1][mask[:, :, 0] == 255] = image_HLS[:, :, 1][mask[:, :,
                                                                  0] == 255] * 0.5  # if red channel is hot, image's "Lightness" channel's brightness is lowered
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
    # Conversion to RGB
    return image_RGB


def buildCar_type1(license_img, num):
    car = cv2.imread('./img/cartype345/car4.jpg')
    plate = cv2.imread(license_img)

    # Getting the coordinates of corners from the first image
    height, width = car.shape[:2]
    h1, w1 = plate.shape[:2]

    pts1 = np.float32([[0, 0], [0, h1], [w1, 0], [w1, h1]])
    pts2 = np.float32([[160, 237], [163, 271], [222, 244], [226, 279]])

    # h is the homography matrix
    h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    height, width, channels = car.shape
    im1Reg = cv2.warpPerspective(plate, h, (width, height))

    # mask filled with same shape as car inn zeros
    mask2 = np.zeros(car.shape, dtype=np.uint8)
    # cv2.imwrite('./img/test_car/mask2.png', mask2)

    # region of interest
    roi_corners = np.int32([[160, 237], [163, 271], [226, 279], [222, 244]])

    channel_count = car.shape[2]
    ignore_mask_color = (255,) * channel_count

    cv2.fillConvexPoly(mask2, roi_corners, ignore_mask_color)

    # not mask and then inverse
    mask2 = cv2.bitwise_not(mask2)
    masked_image = cv2.bitwise_and(car, mask2)
    # cv2.imwrite('./img/test_car/mask2.png', mask2)
    # cv2.imwrite('./img/test_car/masked_image2.png', masked_image)

    # Using Bitwise or to merge the two images
    final = cv2.bitwise_or(im1Reg, masked_image)
    # final = sun_flare(final, flare_center=-1, no_of_flare_circles=8, src_radius=400, src_color=(255, 255, 255))
    final = darken(final)
    cv2.imwrite('./generated_car/test_car/addplate' + str(num) + '.png', final)


def buildCar_type2(license_img, num):
    count = 0

    car = cv2.imread('./img/cartype167/car3.jpg')
    plate = cv2.imread(license_img)

    # Getting the coordinates of corners from the first image
    height, width = car.shape[:2]
    h1, w1 = plate.shape[:2]

    pts1 = np.float32([[0, 0], [0, h1], [w1, 0], [w1, h1]])
    # pts2 = np.float32([[106, 232], [106, 251], [174, 240], [173, 259]])
    pts2 = np.float32([[328, 320], [335, 388], [743, 318], [728, 386]])

    # h is the homography matrix
    h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    height, width, channels = car.shape
    im1Reg = cv2.warpPerspective(plate, h, (width, height))

    # mask filled with same shape as car inn zeros
    mask2 = np.zeros(car.shape, dtype=np.uint8)
    # cv2.imwrite('./img/test_car/mask2.png', mask2)

    # region of interest
    # roi_corners = np.int32([[106, 232], [106, 251], [173, 259], [174, 240]])
    roi_corners = np.int32([[328, 320], [335, 388], [728, 386], [743, 318]])

    channel_count = car.shape[2]
    ignore_mask_color = (255,) * channel_count

    cv2.fillConvexPoly(mask2, roi_corners, ignore_mask_color)

    # not mask and then inverse
    mask2 = cv2.bitwise_not(mask2)
    masked_image = cv2.bitwise_and(car, mask2)
    # cv2.imwrite('./img/test_car/mask2.png', mask2)
    # cv2.imwrite('./img/test_car/masked_image2.png', masked_image)

    # Using Bitwise or to merge the two images
    final = cv2.bitwise_or(im1Reg, masked_image)
    # final = sun_flare(final, flare_center=-1, no_of_flare_circles=8, src_radius=400, src_color=(255, 255, 255))
    # final = darken(final)
    cv2.imwrite('./generated_car/test_car/addplate' + str(num) + '.png', final)


# saved in test_car
buildCar_type1('license1.jpg', "00")
buildCar_type2('license.jpg', "0")

# num = 0
# for file in os.listdir("../data_generate_license/generated_plate/test_plate/"):
#     num += 1
#     file_name = "../data_generate_license/generated_plate/test_plate/" + file
#     plate = cv2.imread(file_name)
#     h = plate.shape[0]
#     # type 1 is the old kind of plates / 2 is the recent plates
#     if h == 170:
#         buildCar_type1(file_name, num)
#     elif h == 110:
#         buildCar_type2(file_name, num)
#     else:
#         # 355x155
#         continue
