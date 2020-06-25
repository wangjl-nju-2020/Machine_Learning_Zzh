class OvR:
    """策略：一对多"""

    def __init__(self, data_set, class_cnt, num_per_class):
        """
        :param data_set: 数据集
        :param class_cnt: 分类个数
        :param num_per_class: 每个分类的数据量个数列表
        """
        self.data_set = data_set
        self.class_cnt = class_cnt
        self.num_per_class = num_per_class
        self.res_data = []

    def divide(self):
        """按'一对多'策略划分数据集合"""
        for i in range(self.class_cnt):
            start = 0
            for j in range(i + 1):
                start += self.num_per_class[j]

            end = start + self.num_per_class

            tmp = [(self.data_set[start:end]), [self.data_set[:start] + self.data_set[end:]]]
            self.res_data.append(tmp)

    def get_divided_data(self):
        self.devide()
        return self.res_data
