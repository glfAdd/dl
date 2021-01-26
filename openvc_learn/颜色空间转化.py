"""
HSV: 这个模型包括色调/饱和度/明度
色调 H: 取值范围 0 度~ 360 度. 0 度位红色, 120 度位绿色, 240 度为蓝色
饱和度 S: 表示颜色接近光谱的程度, 一种颜色可以看做是某种光谱色和白色的混合, 其中光谱占比越大, 颜色越接近光谱本身的颜色
        , 即饱和度越高, 颜色越深, 取值范围 0% - 100%
明度 V: 表示颜色明亮程度, 取值范围 0% - 100%



"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image(image_path: str) -> cv2:
    """读取图片"""
    image_file = cv2.imread(image_path)
    image_file = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)
    return image_file


def show_image(image_file: cv2) -> None:
    """展示图片"""
    plt.axis('off')
    plt.imshow(image_file)
    plt.show()


def split_rgb_image(image_file: cv2) -> None:
    """
    将图片拆成 r g b 并风别展示
    :param image_file:
    :return:
    """
    r, g, b = cv2.split(image_file)
    zero_image = np.zeros(image_file.shape[:2], dtype=np.uint8)
    show_image(cv2.merge([r, zero_image, zero_image]))
    show_image(cv2.merge([zero_image, g, zero_image]))
    show_image(cv2.merge([zero_image, zero_image, b]))


if __name__ == '__main__':
    image = read_image('./static/xing_tai_xue.jpg')
    split_rgb_image(image)
