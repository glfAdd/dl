"""
卷积
卷积窗口:
特征图: 图片求卷积得到的

核: 也叫结构元素, 可是是矩形/椭圆/十字形
"""

import cv2
import matplotlib.pyplot as plt


def get_kernel():
    """获取核心"""
    # 5*5 的矩形核/椭圆/十字
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    # print(kernel_1)
    return kernel_1


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


def fu_shi_image(image_file: cv2) -> None:
    """
    腐蚀: 在原图的小区域内取局部最小值. 腐蚀的次数越多, 图片深色部分越多
    :param image_file:
    :return:
    """
    kernel = get_kernel()
    # iterations 腐蚀次数
    for i in range(3):
        new_image = cv2.erode(image_file, kernel, iterations=i + 1)
        show_image(new_image)


def peng_zhang_image(image_file: cv2) -> None:
    """
    与腐蚀相反, 局部区域取最大值. 膨胀次数越多白色部分越多
    :param image_file:
    :return:
    """
    kernel = get_kernel()
    for i in range(3):
        new_image = cv2.dilate(image_file, kernel, iterations=i + 1)
        show_image(new_image)


def kai_yun_san_image(image_file: cv2) -> None:
    """
    开运算: 先腐蚀后膨胀, 可以消除小白点
    :param image_file:
    :return:
    """
    kernel = get_kernel()
    new_image = cv2.morphologyEx(image_file, cv2.MORPH_OPEN, kernel)
    show_image(new_image)


def bi_yun_suan_image(image_file: cv2) -> None:
    """
    闭运算: 先膨胀后腐蚀, 可以消除小黑点
    :param image_file:
    :return:
    """
    kernel = get_kernel()
    new_image = cv2.morphologyEx(image_file, cv2.MORPH_CLOSE, kernel)
    show_image(new_image)


def ti_du_image(image_file: cv2) -> None:
    """
    gradient 形态学梯度: 膨胀图减去腐蚀图, 得到物体的轮廓
    :param image_file:
    :return:
    """
    kernel = get_kernel()
    new_image = cv2.morphologyEx(image_file, cv2.MORPH_GRADIENT, kernel)
    show_image(new_image)


def bai_mao_image(image_file: cv2) -> None:
    """
    Top Hat (顶帽)/ White Hat (白帽): 原图减去开运算图. 得到元素图片被去掉的白色部分
    :param image_file:
    :return:
    """
    kernel = get_kernel()
    new_image = cv2.morphologyEx(image_file, cv2.MORPH_TOPHAT, kernel)
    show_image(new_image)


def hei_mao_image(image_file: cv2) -> None:
    """
    Black Hat (黑帽): 闭运算图减去原图. 得到被去除掉的黑色部分
    :param image_file:
    :return:
    """
    kernel = get_kernel()
    new_image = cv2.morphologyEx(image_file, cv2.MORPH_BLACKHAT, kernel)
    show_image(new_image)


def averaging_mo_hu_image(image_file: cv2) -> None:
    """
    图像平滑的方式 - averaging 平均: 计算卷积区域内所有像素的平均值得到卷积结果
    卷积窗口越大, 得到的图片越模糊
    """
    kernel = (2, 2)
    kernel = (5, 5)
    kernel = (10, 10)
    # 图片/核心
    new_image = cv2.blur(image_file, kernel)
    show_image(new_image)


def gao_si_mo_hu(image_file: cv2) -> None:
    """
    图像平滑的方式 - 高斯模糊: 卷积时使用的核实高斯核, 核中的值符合高斯分布, 方框中心值最大, 其余位置离中心越远值越小
    卷积窗口越大, 得到的图片越模糊
    """
    # 核心必须是奇数
    # kernel = (2, 2) # 这个高斯的核不能使用
    kernel = (5, 5)
    kernel = (15, 15)
    # 图片/核心/标准差
    new_image = cv2.GaussianBlur(image_file, kernel, 0)
    show_image(new_image)


def zhong_zhi_image(image_file: cv2) -> None:
    """
    图像平滑的方式 - Median 中值模糊: 用卷积框像素的中值代替中心像素值
    :param image_file:
    :return:
    """
    # 核心必须是奇数
    kernel = 3
    kernel = 5
    kernel = 9
    # 图片/核心/标准差
    new_image = cv2.medianBlur(image_file, kernel)
    show_image(new_image)


def shuang_bian_lu_bo_image() -> None:
    """
    图像平滑的方式 - Bilateral 双边滤波: 在保持清晰度的情况下有效去除噪音.
    高斯滤波器值考虑像素之间的空间关系, 而不会考虑像素值之间的关系(像素的相似度), 所以这个方法不会考虑像素是否位于边界, 所以边界会模糊

    双边滤波同事使用空间高斯权重和灰度值相似高斯权重, 空间高斯函数确保只有邻近区域的像素对中心点有影响, 灰度值相似性高斯函数确保之后与中心像素灰度值相近的才会被用来做模糊运算.
    所以这种方法确保边界不会被模糊, 因为边界处的灰度值变化比较大
    """
    # 临域直径/灰度值相似性高斯函数标准差/空间高斯函数标准差
    image_file = read_image('./static/wood.jpg')
    new_image_1 = cv2.bilateralFilter(image_file, 10, 10, 10)
    show_image(new_image_1)
    # new_image_2 = cv2.bilateralFilter(image_file, 50, 50, 50)
    # show_image(new_image_2)


if __name__ == '__main__':
    path = './static/xing_tai_xue.jpg'
    image = read_image(path)
    # fu_shi_image(image)
    # peng_zhang_image(image)
    # kai_yun_san_image(image)
    # bi_yun_suan_image(image)
    # ti_du_image(image)
    # bai_mao_image(image)
    # hei_mao_image(image)
    # averaging_mo_hu_image(image)
    # gao_si_mo_hu(image)
    # zhong_zhi_image(image)
    shuang_bian_lu_bo_image()
