from math import exp
from math import log

import numpy as np

from exercise03.utils import *


class GradientDescent:
    """递归下降法"""

    def __init__(self, attr_matrix, matrix_r, matrix_c):
        """
        :param attr_matrix: 数据集矩阵
        :param matrix_r: 矩阵行数
        :param matrix_c: 矩阵列数
        """
        self.attr_matrix = attr_matrix
        self.matrix_r = matrix_r
        self.matrix_c = matrix_c
        self.beta = []
        self.init_beta()

    def init_beta(self):
        for i in range(self.matrix_c - 1):
            self.beta.append(0)
        self.beta.append(1)

    def func_beta(self):
        """目标函数"""
        func_v = 0

        for x in self.attr_matrix:
            beta_x = self.beta[self.matrix_c - 1]
            for i in range(self.matrix_c - 1):
                beta_x += self.beta[i] * x[i]

            func_v += -x[self.matrix_c - 1] * beta_x + log(1 + exp(beta_x))

        return func_v

    def grad(self):
        """计算梯度"""
        deri = []
        for i in range(self.matrix_c):
            deri.append(0.0)

        for x in self.attr_matrix:
            beta_x = self.beta[self.matrix_c - 1]
            for i in range(self.matrix_c - 1):
                beta_x += self.beta[i] * x[i]

            for i in range(self.matrix_c - 1):
                deri[i] += x[i] * (-x[self.matrix_c - 1] + 1 - 1 / (1 + exp(beta_x)))

            deri[self.matrix_c - 1] += -x[self.matrix_c - 1] + 1 - 1 / (1 + exp(beta_x))

        return deri

    def grad_desc(self, lr=0.01, prec=0.0001, max_iters=1000000):
        """
        梯度下降
        :param lr: 学习率
        :param prec: 精度
        :param max_iters: 最大迭代次数
        :return:
        """
        print("------start gradient_descent-----------")

        for epoch in range(max_iters):
            grad = self.grad()
            grad_len = np.linalg.norm(grad)
            if grad_len < prec:
                print("------end gradient_descent-----------")
                print("拟合完成")
                print("grad_len = %.5f" % grad_len)
                print("grad = ", end='')
                print_formatted_vec(grad, self.matrix_c)
                break

            for i in range(self.matrix_c):
                self.beta[i] -= lr * grad[i]

            print("第%d轮迭代beta值为" % epoch, end='')
            print_formatted_vec(self.beta, self.matrix_c)

        return self.beta, self.func_beta()


class Model:
    """对数几率模型"""

    def __init__(self, beta, attr_len):
        self.beta = beta
        self.attr_len = attr_len

    def predict(self, x):
        y = 0
        for i in range(self.attr_len):
            y += x[i] * self.beta[i]

        if y > 0.5:
            return 1
        else:
            return 0
