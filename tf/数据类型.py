"""
***************************************** 数据类型
有符号整型
    tf.int8：8位整数。
    tf.int16：16位整数。
    tf.int32：32位整数。
    tf.int64：64位整数。
无符号整型
    tf.uint8：8位无符号整数。
    tf.uint16：16位无符号整数。
浮点型
    tf.float16：16位浮点数。
    tf.float32：32位浮点数。
    tf.float64：64位浮点数。
    tf.double：等同于tf.float64。
字符串型
    tf.string：字符串。
布尔型
    tf.bool：布尔型。
复数型
    tf.complex64：64位复数。
    tf.complex128：128位复数。

***************************************** Tensorflow Tensor的属性
device	设备位置
numpy	numpy数据
shape	数据形状
ndim	数据维度
rank	数据维度(返回tensor类型)
name	名称（tf2.0无效了）

"""

import tensorflow as tf
import numpy as np

print(""" ================================== 版本和是否可用""")
print(tf.__version__)
print(tf.test.is_gpu_available())

print(""" ================================== 设置常量""")
a_1 = tf.constant(1)
a_2 = tf.constant(2.2, dtype=tf.double)
a_3 = tf.constant('Hello Word', dtype=tf.string)

print(""" ================================== 选择运行的设备""")
# tensorflow中不同的GPU使用/gpu:0和/gpu:1区分，而CPU不区分设备号，统一使用 /cpu:0
with tf.device('/cpu:1'):
    b_1 = tf.range(8)

with tf.device('/gpu:2'):
    b_2 = tf.constant(1)

print(b_1.device)
print(b_2.device)

print(""" ================================== 切换运行的设备""")
print(b_1.gpu().device)
print(b_2.cpu().device)

print(""" ================================== 判断类型""")
print(tf.is_tensor(b_1))
print(isinstance(b_1, tf.Tensor))
print(b_1.dtype)

print(""" ================================== 类型转换 """)
d_1 = np.arange(9)

# np 转为 tf
# convert_to_tensor 参数可以是python list/ones/zeros/numpy array
d_2 = tf.convert_to_tensor(d_1)
d_3 = tf.convert_to_tensor(d_1, dtype=tf.int64)

# tf 转为 np
d_5 = d_2.numpy()
print(d_1.dtype, d_2.dtype, d_3.dtype, d_5.dtype)

# tf 与 tf 类型装换
d_4 = tf.cast(d_3, dtype=tf.float64)
print(d_4)

print(""" ================================== Variables """)
