"""
西瓜书练习3.3，使用梯度下降法
"""

import matplotlib.pyplot as plt
import numpy as np

import exercise03.data_loader as data_loader
import exercise03.my_data_set as mds
from exercise03.gradient_descent import GradientDescent
from exercise03.utils import *


# def get_formatted_data(ds):
#     """从数据集得到格式化的数据"""
#     data = []
#     for d in ds:
#         den = d['density']
#         sug = d['sugar']
#         flag = d['flag']
#         if flag:
#             y = 1
#         else:
#             y = 0
#         data.append([den, sug, y])
#     return data
#
#
# def func_beta(data, beta):
#     """
#     计算目标函数值
#     :param data: 数据集
#     :param beta: 参数向量beta
#     :return: 函数值
#     """
#     func_v = 0
#     for d in data:
#         beta_x = beta[0] * d[0] + beta[1] * d[1] + beta[2]
#         func_v += -d[2] * beta_x + log(1 + exp(beta_x))
#
#     return func_v
#
#
# def grad_f(data, beta):
#     """
#     计算梯度
#     :param data: 数据集
#     :param beta: 参数向量beta
#     :return: 梯度
#     """
#     deri_0 = 0
#     deri_1 = 0
#     deri_2 = 0
#
#     for d in data:
#         beta_x = beta[0] * d[0] + beta[1] * d[1] + beta[2]
#         deri_0 += d[0] * (-d[2] + 1 - 1 / (1 + exp(beta_x)))
#         deri_1 += d[1] * (-d[2] + 1 - 1 / (1 + exp(beta_x)))
#         deri_2 += -d[2] + 1 - 1 / (1 + exp(beta_x))
#
#     return [deri_0, deri_1, deri_2]
#
#
# def gradient_descent_f(data, cur_beta, lr=0.01, prec=0.0001, max_iters=1000000):
#     """
#     梯度下降法
#     :param data: 数据集
#     :param cur_beta: 当前参数向量cur_beta
#     :param lr: 学习率
#     :param prec: 精度
#     :param max_iters: 最大迭代次数
#     :return: 最终参数向量
#     """
#     print("------start gradient_descent-----------")
#
#     for epoch in range(max_iters):
#         grad = grad_f(data, cur_beta)
#         grad_len = sqrt(grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2])
#         if grad_len < prec:
#             print("------end gradient_descent-----------")
#             print("拟合完成，grad_len = %.5f" % grad_len)
#             break
#
#         for i in range(3):
#             cur_beta[i] = cur_beta[i] - lr * grad[i]
#
#         print("第%d轮迭代beta值为(%.4f, %.4f, %.4f), grad_len = %.4f"
#               % (epoch, cur_beta[0], cur_beta[1], cur_beta[2], grad_len))
#
#     print("最终beta值为(%.4f, %.4f, %.4f)"
#           % (cur_beta[0], cur_beta[1], cur_beta[2]))
#
#     return cur_beta


def show(data, beta):
    """
    绘制散点以及分界线
    :param data: 数据集
    :param beta: 参数向量
    :return:
    """
    plt.title('Logistic Regression')
    plt.xlabel('density'.title())
    plt.ylabel('sugar rate'.title())

    # 计算散点
    x_values = [d[0] for d in data]
    y_values = [d[1] for d in data]
    # 绘制正例
    ps = plt.scatter(x_values[0:8], y_values[0:8], c='blue', marker='o')
    # 绘制反例
    ns = plt.scatter(x_values[8:], y_values[8:], c='red', marker='^')
    # 图例
    plt.legend((ps, ns), ('positive sample', 'negative sample'))

    # 计算分界线上的点：w1*x1+w2*x2+b=0.5
    x1 = np.arange(0, 1, 0.01)
    x2 = (0.5 - x1 * beta[0] - beta[2]) / beta[1]
    # 绘制分界线
    plt.plot(x1, x2)

    plt.savefig('logistic_regression.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    dl3 = data_loader.Demo33DataLoader()
    dl3.load_data()
    data = dl3.get_formatted_data()
    print(data)

    gd = GradientDescent(data, 17, 3)
    beta, func_v = gd.grad_desc()

    print('beta = ', end='')
    print_formatted_vec(beta, 3)

    print('func_v = %.5f' % func_v)
    show(data, beta)
