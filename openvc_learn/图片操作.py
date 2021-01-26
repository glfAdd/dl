import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image(image_path: str) -> cv2:
    """
    读取图片
    :param image_path: 图片路径
    :return:
    """
    file = cv2.imread(image_path)
    # 设置图片通道顺序, matplotlib 打开图片和实际的图片颜色不一样, 因为颜色通道的顺序不一样, BGR 转为 RGB
    file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
    return file


def show_image(image_file: cv2) -> None:
    """
    展示图片
    :param image_file:
    :return:
    """
    # 关闭图片的坐标系
    plt.axis('off')
    plt.imshow(image_file)
    plt.show()


def save_image(image_file: cv2, image_name: str) -> None:
    """
    保存图片
    :param image_file: 图片文件
    :param image_name: 图片名称
    :return:
    """
    save = cv2.imwrite(image_name, image_file)
    print(save)


def split_image(image_file: cv2) -> None:
    """
    分割图片
    :param image_file:
    :return:
    """
    # 高 宽 通道数量
    h, w, s = image_file.shape
    ch = h // 2
    cw = w // 2
    image_1 = image[:ch, :cw]
    image_2 = image[ch:h, cw: w]
    show_image(image_1)
    show_image(image_2)


def change_one_color(image_file: cv2) -> None:
    """
    改变图片一个像素点的颜色
    :param image_file:
    :return:
    """
    image_file[100, 20] = (0, 255, 0)
    print(image_file[100, 20])


def change_part_color(image_file: cv2) -> None:
    """
    改变图片一个区域的颜色
    :param image_file:
    :return:
    """
    image_file[100:400, 400:800] = (255, 0, 0)
    show_image(image_file)
    pass


def move_image(image_file: cv2) -> None:
    """
    移动图片
    :param image_file:
    :return:
    """
    # 1,0 表示水平方向
    # 0,1 表示数值方向
    m = np.float32([[1, 0, 400], [0, 1, -200]])
    size = (image_file.shape[1], image_file.shape[0])
    new_image = cv2.warpAffine(image_file, m, size)
    show_image(new_image)


def turn_image(image_file: cv2) -> None:
    """
    旋转图片
    :param image_file:
    :return:
    """
    # 索引从 1 开始
    h, w = image_file.shape[:2]
    print(h, w)
    # 中心点, 旋转角度, 缩放比例
    m = cv2.getRotationMatrix2D((h // 2, w // 2), 45, 1.0)
    new_image = cv2.warpAffine(image_file, m, (w, h))
    show_image(new_image)


def resize_image(image_file: cv2) -> None:
    """
    改变图片尺寸, 同时改变比例
    :param image_file:
    :return:
    """
    w, h = 500, 200
    # 有五种计算的算法
    # 最邻近
    # new_image = cv2.resize(image_file, (w, h), interpolation=cv2.INTER_NEAREST)
    # 双线性
    # new_image = cv2.resize(image_file, (w, h), interpolation=cv2.INTER_LINEAR)
    # 基于像素区域
    # new_image = cv2.resize(image_file, (w, h), interpolation=cv2.INTER_AREA)
    # 立方插值
    # new_image = cv2.resize(image_file, (w, h), interpolation=cv2.INTER_CUBIC)
    # 兰索斯插值
    new_image = cv2.resize(image_file, (w, h), interpolation=cv2.INTER_LANCZOS4)
    show_image(new_image)


def print_line() -> None:
    """ 画线 """
    # 用 numpy 创建图片
    image_file = np.zeros((300, 300, 3), dtype='uint8')

    # 直线. 起点/终点/颜色/线宽
    cv2.line(image_file, (0, 0), (300, 300), (255, 0, 0), 1)
    cv2.line(image_file, (300, 0), (0, 300), (0, 255, 0), 5)

    # 矩形. 左上角/右下角/颜色/线宽
    cv2.rectangle(image_file, (20, 20), (50, 50), (0, 0, 255), 3)
    # -1 表示填充
    cv2.rectangle(image_file, (120, 120), (50, 50), (0, 0, 255), -1)

    # 圆. 图片/圆心/半径/颜色,线宽
    cv2.circle(image_file, (200, 200), 30, (255, 255, 255), 2)
    show_image(image_file)


def mirror_image(image_file: cv2) -> None:
    """
    镜像翻转图片
    :param image_file:
    :return:
    """
    # 1 y轴翻转,
    # 2 x 轴翻转
    # -1 x和 y 同时翻转
    new_image = cv2.flip(image_file, -1)
    show_image(new_image)


def count_image(image_file: cv2) -> None:
    """
    图片运算
    加法: 图片通道最大值为 255, 继续增大也是 255
    减法: 图片通道最小值是 0, 继续减小时也是 0

    uint8 普通运算
    加法: uint8 最大值为 255, 继续增加时会从 0 重新增加
    减法: uint8 最小值为 0 , 继续减小时会送 255 重新减小
    """
    # 创建时值在 0-255 之间
    a = np.uint8([70])
    b = np.uint8([200])
    print(cv2.add(a, b))
    print(cv2.subtract(a, b))

    print(a + b)
    print(a - b)

    # 生成和原图片尺寸相同, 所有值都是 100 的数据
    m = np.ones(image_file.shape, dtype=np.uint8) * 100
    # 图片所有通道增加 100
    new_image = cv2.add(image_file, m)
    show_image(new_image)


def bit_count_image(image_file: cv2) -> None:
    """
    按位运算
    与       1&1=1, 1&0=0, 0&1=0, 0&0=0
    或       1|1=1, 1|0=1, 0|1=1, 0|0=0
    异或      1^1=0, 1^0=1, 0^1=1, 0^0=0
    非       ~0=1, ~1=0

    """
    image_1 = np.zeros((300, 300, 3), dtype=np.uint8)
    image_1 = cv2.rectangle(image_1, (25, 25), (275, 275), (255, 255, 255), -1)

    image_2 = np.zeros((300, 300, 3), dtype=np.uint8)
    image_2 = cv2.circle(image_2, (150, 150), 150, (255, 255, 255), -1)

    # 与, 有黑变黑
    image_and = cv2.bitwise_and(image_1, image_2)
    # 或, 有白变白
    image_or = cv2.bitwise_or(image_1, image_2)
    # 异或, 黑白变白, 黑黑或白白变黑
    image_xor = cv2.bitwise_xor(image_1, image_2)
    # 非, 取反
    image_not = cv2.bitwise_not(image_2)
    show_image(image_not)


def bit_use_image(image_file) -> None:
    """
    使用按位运算实现遮挡
    :param image_file:
    :return:
    """
    top_image = np.zeros(image_file.shape, dtype=np.uint8)
    top_image = cv2.rectangle(top_image, (25, 25), (1000, 1000), (255, 255, 255), -1)
    new_image = cv2.bitwise_and(top_image, image_file)
    show_image(new_image)


def split_merge_image(image_file) -> None:
    # 将图片的三个通道拆分开
    r, g, b = cv2.split(image_file)
    print(r.shape, g.shape, b.shape)
    show_image(b)

    # 和并三个通道成为一个图片
    new_image = cv2.merge([r, g, b])


def gao_si_image(image_file: cv2) -> None:
    """
    高斯金字塔图像
    :param image_file:
    :return:
    """
    for i in range(4):
        image_file = cv2.pyrDown(image_file)
        show_image(image_file)
        print(image_file.shape)

    for i in range(4):
        image_file = cv2.pyrUp(image_file)
        show_image(image_file)
        print(image_file.shape)


def la_pu_la_si_image(image_file: cv2) -> None:
    """
    拉普拉斯金字塔
    操作步骤:
        1. 降低一次分辨率
        2. 降低一次分辨率
        3. 提升一次分辨率
        4. 用第1步的图片减去第3部的图片, 得到的就是拉普拉斯金字塔
    :param image_file:
    :return:
    """
    image_1 = cv2.pyrDown(image_file)
    image_2 = cv2.pyrDown(image_1)
    image_3 = cv2.pyrUp(image_2)
    last = image_1 - image_3
    show_image(last)


if __name__ == '__main__':
    file_path = './static/test.jpg'
    image = read_image(file_path)
    # show_image(image)
    # split_image(image)
    # change_one_color(image)
    # change_part_color(image)
    # move_image(image)
    # turn_image(image)
    # resize_image(image)
    # print_line()
    # mirror_image(image)
    # count_image(image)
    # bit_count_image(image)
    # bit_use_image(image)
    # split_merge_image(image)
    # gao_si_image(image)
    # la_pu_la_si_image(image)
    la_pu_la_si_image(image)
