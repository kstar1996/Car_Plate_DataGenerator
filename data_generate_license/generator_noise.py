import math
import os, random
import cv2, argparse
import numpy as np


# Sunny
def change_brightness(image):
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)  # Conversion to HLS
    image_HLS = np.array(image_HLS, dtype=np.float64)
    random_brightness_coefficient = np.random.uniform() + 0.5  # generates value between 0.5 and 1.5
    image_HLS[:, :, 1] = image_HLS[:, :, 1] * random_brightness_coefficient  # scale pixel values up or down for channel 1(Lightness)
    image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255  # Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype=np.uint8)
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)  # Conversion to RGB
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


def augmentation(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = sun_flare(img, flare_center=-1, no_of_flare_circles=8, src_radius=400, src_color=(255, 255, 255))
    return img


class ImageGenerator:
    def __init__(self, save_path):
        self.save_path = save_path
        # Plate
        self.plate = cv2.imread("img/plate.jpg")
        self.plate2 = cv2.imread("img/plate_y.jpg")
        self.plate3 = cv2.imread("img/plate_g.jpg")
        self.plate4 = cv2.imread("img/plate_e.jpg")

        # loading Number
        file_path = "img/num/"
        file_list = os.listdir(file_path)
        self.Number = list()
        self.number_list = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Number.append(img)
            self.number_list.append(file[0:-4])

        # loading Char
        file_path = "img/char1/"
        file_list = os.listdir(file_path)
        self.char_list = list()
        self.Char1 = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Char1.append(img)
            self.char_list.append(file[0:-4])

        # loading Number ====================  blue-one-line  ==========================
        file_path = "img/num_e/"
        file_list = os.listdir(file_path)
        self.Number_e = list()
        self.number_list_e = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Number_e.append(img)
            self.number_list_e.append(file[0:-4])

        # loading Char
        file_path = "img/char1_e/"
        file_list = os.listdir(file_path)
        self.char_list_e = list()
        self.Char1_e = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Char1_e.append(img)
            self.char_list_e.append(file[0:-4])
        # =========================================================================

        # loading Number ====================  yellow-two-line  ==========================
        file_path = "img/num_y/"
        file_list = os.listdir(file_path)
        self.Number_y = list()
        self.number_list_y = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Number_y.append(img)
            self.number_list_y.append(file[0:-4])

        # loading Char
        file_path = "img/char1_y/"
        file_list = os.listdir(file_path)
        self.char_list_y = list()
        self.Char1_y = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Char1_y.append(img)
            self.char_list_y.append(file[0:-4])

        # loading Resion
        file_path = "img/region_y/"
        file_list = os.listdir(file_path)
        self.Resion_y = list()
        self.resion_list_y = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Resion_y.append(img)
            self.resion_list_y.append(file[0:-4])
        # =========================================================================

        # loading Number ====================  green-two-line  ==========================
        file_path = "img/num_g/"
        file_list = os.listdir(file_path)
        self.Number_g = list()
        self.number_list_g = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Number_g.append(img)
            self.number_list_g.append(file[0:-4])

        # loading Char
        file_path = "img/char1_g/"
        file_list = os.listdir(file_path)
        self.char_list_g = list()
        self.Char1_g = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Char1_g.append(img)
            self.char_list_g.append(file[0:-4])

        # loading Resion
        file_path = "img/region_g/"
        file_list = os.listdir(file_path)
        self.Resion_g = list()
        self.resion_list_g = list()
        for file in file_list:
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            self.Resion_g.append(img)
            self.resion_list_g.append(file[0:-4])
        # =========================================================================

    def Type_1(self, num, save=False):
        number = [cv2.resize(number, (56, 83)) for number in self.Number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char1]
        Plate = cv2.resize(self.plate, (520, 110))

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (520, 110))
            label = "Z"
            # row -> y , col -> x
            row, col = 13, 35  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # character 3
            label += self.char_list[i % 37]
            Plate[row:row + 83, col:col + 60, :] = char[i % 37]
            col += (60 + 36)

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56
            Plate = augmentation(Plate)
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Type_2(self, num, save=False):
        number = [cv2.resize(number, (45, 83)) for number in self.Number]
        char = [cv2.resize(char1, (49, 70)) for char1 in self.Char1]
        Plate = cv2.resize(self.plate, (355, 155))

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (355, 155))
            label = "Z"
            row, col = 46, 10  # row + 83, col + 56

            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 45, :] = number[rand_int]
            col += 45

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 45, :] = number[rand_int]
            col += 45

            # number 3
            label += self.char_list[i % 37]
            Plate[row + 12:row + 82, col + 2:col + 49 + 2, :] = char[i % 37]
            col += 49 + 2

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col + 2:col + 45 + 2, :] = number[rand_int]
            col += 45 + 2

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 45, :] = number[rand_int]
            col += 45

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 45, :] = number[rand_int]
            col += 45

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 45, :] = number[rand_int]
            col += 45
            Plate = augmentation(Plate)
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Type_3(self, num, save=False):
        number1 = [cv2.resize(number, (44, 60)) for number in self.Number_y]
        number2 = [cv2.resize(number, (64, 90)) for number in self.Number_y]
        resion = [cv2.resize(resion, (88, 60)) for resion in self.Resion_y]
        char = [cv2.resize(char1, (64, 62)) for char1 in self.Char1_y]

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate2, (336, 170))

            label = str()
            # row -> y , col -> x
            row, col = 8, 76

            # resion
            label += self.resion_list_y[i % 16]
            Plate[row:row + 60, col:col + 88, :] = resion[i % 16]
            col += 88 + 8

            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            Plate[row:row + 60, col:col + 44, :] = number1[rand_int]
            col += 44

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            Plate[row:row + 60, col:col + 44, :] = number1[rand_int]

            row, col = 72, 8

            # character 3
            label += self.char_list_y[i % 37]
            Plate[row:row + 62, col:col + 64, :] = char[i % 37]
            col += 64

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
            col += 64

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
            col += 64

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
            col += 64

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list_y[rand_int]
            Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
            Plate = augmentation(Plate)
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Type_4(self, num, save=False):
        number1 = [cv2.resize(number, (44, 60)) for number in self.Number_g]
        number2 = [cv2.resize(number, (64, 90)) for number in self.Number_g]
        resion = [cv2.resize(resion, (88, 60)) for resion in self.Resion_g]
        char = [cv2.resize(char1, (64, 62)) for char1 in self.Char1_g]

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate3, (336, 170))

            label = str()
            # row -> y , col -> x
            row, col = 8, 76

            # resion
            label += self.resion_list_g[i % 16]
            Plate[row:row + 60, col:col + 88, :] = resion[i % 16]
            col += 88 + 8

            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            Plate[row:row + 60, col:col + 44, :] = number1[rand_int]
            col += 44

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            Plate[row:row + 60, col:col + 44, :] = number1[rand_int]

            row, col = 72, 8

            # character 3
            label += self.char_list_g[i % 37]
            Plate[row:row + 62, col:col + 64, :] = char[i % 37]
            col += 64

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
            col += 64

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
            col += 64

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
            col += 64

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            Plate[row:row + 90, col:col + 64, :] = number2[rand_int]
            Plate = augmentation(Plate)
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Type_5(self, num, save=False):
        number1 = [cv2.resize(number, (60, 65)) for number in self.Number_g]
        number2 = [cv2.resize(number, (80, 90)) for number in self.Number_g]
        char = [cv2.resize(char1, (60, 65)) for char1 in self.Char1_g]

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate3, (336, 170))
            random_width, random_height = 336, 170
            label = "Z"

            # row -> y , col -> x
            row, col = 8, 78

            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            Plate[row:row + 65, col:col + 60, :] = number1[rand_int]
            col += 60

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            Plate[row:row + 65, col:col + 60, :] = number1[rand_int]
            col += 60

            # character 3
            label += self.char_list_g[i % 37]
            Plate[row:row + 65, col:col + 60, :] = char[i % 37]
            row, col = 75, 8

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            Plate[row:row + 90, col:col + 80, :] = number2[rand_int]
            col += 80

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            Plate[row:row + 90, col:col + 80, :] = number2[rand_int]
            col += 80

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            Plate[row:row + 90, col:col + 80, :] = number2[rand_int]
            col += 80

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list_g[rand_int]
            Plate[row:row + 90, col:col + 80, :] = number2[rand_int]

            Plate = augmentation(Plate)

            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Type_6(self, num, save=False):
        number = [cv2.resize(number, (55, 83)) for number in self.Number]
        char = [cv2.resize(char1, (60, 83)) for char1 in self.Char1]
        Plate = cv2.resize(self.plate, (520, 110))

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate, (520, 110))

            label = ""
            # row -> y , col -> x
            row, col = 13, 30  # row + 83, col + 56

            # number 0
            rand_int = random.randint(1, 6)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 55, :] = number[rand_int]
            col += 55

            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 55, :] = number[rand_int]
            col += 55

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 55, :] = number[rand_int]
            col += 55

            # character 3
            label += self.char_list[i % 37]
            Plate[row:row + 83, col:col + 60, :] = char[i % 37]
            col += (60 + 15)

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 55, :] = number[rand_int]
            col += 55

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 55, :] = number[rand_int]
            col += 55

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 55, :] = number[rand_int]
            col += 55

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list[rand_int]
            Plate[row:row + 83, col:col + 55, :] = number[rand_int]
            col += 55
            Plate = augmentation(Plate)
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def Type_7(self, num, save=False):
        number = [cv2.resize(number_e, (56, 83)) for number_e in self.Number_e]
        char = [cv2.resize(char1_e, (60, 83)) for char1_e in self.Char1_e]
        Plate = cv2.resize(self.plate4, (520, 110))

        for i, Iter in enumerate(range(num)):
            Plate = cv2.resize(self.plate4, (520, 110))
            label = "X"
            # row -> y , col -> x
            row, col = 13, 44  # row + 83, col + 56
            # number 1
            rand_int = random.randint(0, 9)
            label += self.number_list_e[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 2
            rand_int = random.randint(0, 9)
            label += self.number_list_e[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # character 3
            label += self.char_list_e[i % 37]
            Plate[row:row + 83, col:col + 60, :] = char[i % 37]
            col += (60 + 36)

            # number 4
            rand_int = random.randint(0, 9)
            label += self.number_list_e[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 5
            rand_int = random.randint(0, 9)
            label += self.number_list_e[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 6
            rand_int = random.randint(0, 9)
            label += self.number_list_e[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56

            # number 7
            rand_int = random.randint(0, 9)
            label += self.number_list_e[rand_int]
            Plate[row:row + 83, col:col + 56, :] = number[rand_int]
            col += 56
            Plate = augmentation(Plate)
            if save:
                cv2.imwrite(self.save_path + label + ".jpg", Plate)
            else:
                cv2.imshow(label, Plate)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--img_dir", help="save image directory",
                    type=str, default="./generated_plate/test_plate/")
parser.add_argument("-n", "--num", help="number of image",
                    type=int, default=50)
parser.add_argument("-s", "--save", help="save or imshow",
                    type=bool, default=True)
args = parser.parse_args()

img_dir = args.img_dir
A = ImageGenerator(img_dir)
img_dir2 = "./generated_plate/train_plate/"
B = ImageGenerator(img_dir2)

num_img = args.num
Save = args.save

print("test_plate")
A.Type_1(num_img, save=Save)
print("Type 1 finish")
A.Type_2(num_img, save=Save)
print("Type 2 finish")
A.Type_3(num_img, save=Save)
print("Type 3 finish")
A.Type_4(num_img, save=Save)
print("Type 4 finish")
A.Type_5(num_img, save=Save)
print("Type 5 finish")
A.Type_6(num_img, save=Save)
print("Type 6 finish")
A.Type_7(num_img, save=Save)
print("Type 7 finish")

print("train_plate")
B.Type_1(num_img, save=Save)
print("Type 1 finish")
B.Type_2(num_img, save=Save)
print("Type 2 finish")
B.Type_3(num_img, save=Save)
print("Type 3 finish")
B.Type_4(num_img, save=Save)
print("Type 4 finish")
B.Type_5(num_img, save=Save)
print("Type 5 finish")
B.Type_6(num_img, save=Save)
print("Type 6 finish")
B.Type_7(num_img, save=Save)
print("Type 7 finish")
