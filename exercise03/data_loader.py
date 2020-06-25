"""
加载需要的数据集
"""
import exercise03.data_config as dc
import exercise03.my_data_set as mds

data_repo = '~/Downloads/Machine_Learning_Repository'


class DataLoader:

    def __init__(self, file_name=''):
        self.file_name = file_name
        self.dir = data_repo + file_name
        self.data = []

    def load_data(self):
        pass

    def get_formatted_data(self):
        return self.data


class IrisDataLoader(DataLoader):
    """读取iris数据集"""

    def load_data(self):
        with open(self.dir) as f_obj:
            for line in f_obj.read():
                attrs = line.split(',')
                for i in range(4):
                    attrs[i] = float(attrs[i])
                attrs[4] = dc.iris_class_map[attrs[4]]
                self.data.append(attrs)


class Demo33DataLoader(DataLoader):
    """读取练习3.3数据"""

    def load_data(self):
        for d in mds.data_set:
            den = d['density']
            sug = d['sugar']
            flag = d['flag']
            if flag:
                y = 1
            else:
                y = 0
            self.data.append([den, sug, y])
