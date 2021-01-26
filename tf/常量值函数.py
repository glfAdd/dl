"""
tensorflow中的shape参数都是python中下list , 而不能是单个的数字
"""
import tensorflow as tf

"""
def constant(value, dtype=None, shape=None, name="Const"):
    value: 可以是一个数值，也可以是一个列表. 
        如果是一个数，那么这个tensor中所有值的按该数来赋值。 
        如果是list, 那么value的长度一定要等于shape展开后的长度
    dtype: 所要创建的tensor的数据类型
    shape: 表示维度
    name: 该操作的别名
"""
print(""" ================================== constant """)
print(tf.constant(1))
print(tf.constant(2, dtype=tf.float64, shape=(9,)))
print(tf.constant(2, shape=[9]))
print(tf.constant([4, 5, 6], shape=[3]))
print(tf.constant([4, 5, 6, 7, 8, 9], shape=[3, 2]))

"""
每个元素都是0
def zeros(shape, dtype=dtypes.float32, name=None):
    shape:表示维度
    dtype: 数据类型
    name: 该操作的别名
"""
print(""" ================================== zeros """)
print(tf.zeros([2, 5], dtype=tf.float64))
print(tf.zeros((2, 4)))

"""
将原来的元素都替换为 0
"""
print(""" ================================== zeros_like """)
a = tf.constant(6, dtype=tf.int64, shape=[3, 4])
print(a)
print(tf.zeros_like(a))

"""
每个元素都用默认值
def fill(dims, value, name=None):
    dims: list, 维度
    value: 每个元素的默认值
    name: 当前操作别名
"""
print(""" ================================== fill """)
print(tf.fill([2, 3, 4], value='test'))

"""
根据维度创建元素都为 1 的张量
"""
print(""" ================================== ones """)
print(tf.ones(shape=[2, 3, 4], dtype=tf.int32))

"""

"""
print(""" ================================== random """)
# 随机生成张量
random_1 = tf.random.normal([3, 4, 3, 4])

print(random_1)
print(random_1[1].shape)
print(random_1[1, 2].shape)
print(random_1[1, 2, 2].shape)
print(random_1[1, 2, 2])

# print(random_1[1::])
# print(random_1[::0].shape)
# print(random_1[:2:])
# print(random_1[1::1])
