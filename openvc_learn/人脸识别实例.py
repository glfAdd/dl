"""
图像金字塔: 同一个图像不同分辨率的子图像集合.
例如在人脸识别时, 不知道人脸有多大, 识别框能不能框住, 所以需要在不同的分辨率检测

金字塔图像分为两种:
    高斯金字塔: 图像每次变为原来的 1/4
    拉普拉斯金字塔:
"""

import cv2
import matplotlib.pyplot as plt

# image_path = './static/face.png'
image_path = './static/Solvay.jpg'
image_file = cv2.imread(image_path)
image_file = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)

# 级联分类器检测人脸
detector = cv2.CascadeClassifier('./static/haarcascade_frontalface_default.xml')
"""
def detectMultiScale(self, image, scaleFactor=None, minNeighbors=None, flags=None, minSize=None, maxSize=None): 
    image: cv2 图片对象
    scaleFactor: 每次缩小图像比例, 默认 1.1
    minNeighbors: 匹配成功需要周围矩形框的数量, 每个特征匹配到的区域都是一个矩形框, 只有多个矩形框同时存在时才认为是匹配成功, 比如人脸默认是 3
    flags: 
        CASCADE_DO_CANNY_PRUNING: 利用 canny边缘检测来排除一些边缘很少或很多的图像区域
        CASCADE_SCALE_IMAGE: 正常比例检测
        CASCADE_FIND_BIGGEST_OBJECT: 只检测最大的物体
        CASCADE_DO_ROUGH_SEARCH: 初略的检测
    minSize: 匹配人脸的最小范围
    maxSize: 
"""
rects = detector.detectMultiScale(image_file, scaleFactor=1.1, minNeighbors=2, minSize=(10, 10),
                                  flags=cv2.CASCADE_SCALE_IMAGE)

# 循环化矩形框
for x, y, w, h in rects:
    # 图片, 左上角, 右下角, 颜色, 线宽度
    cv2.rectangle(image_file, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.axis('off')
plt.imshow(image_file)
plt.show()
