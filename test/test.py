import numpy as np
outputs = [[0.1, 0.4, 0.2, 0.3],
           [0.2, 0.3, 0.5, 0.0],
           [0.4, 0.2, 0.1, 0.3]]


print(np.arange(len(outputs)))

import torch

# 创建一个6x768的全零张量
tensor_shape = (6, 768)
one_hot_tensor = torch.zeros(tensor_shape)

# 选择要设置为1的行和列索引
row_indices = torch.arange(6)  # 0到5，共6个样本
# column_index = torch.randint(0, 768, (6,))  # 随机选择0到767之间的列索引

column_index = torch.arange(6)
# 在one_hot_tensor中将选定的位置设置为1
one_hot_tensor[row_indices, column_index] = 1


# 现在，one_hot_tensor是一个6x768的独热向量张量

if __name__ == '__main__':
    pass