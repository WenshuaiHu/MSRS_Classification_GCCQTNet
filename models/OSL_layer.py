from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch
import numpy as np
import math

def design_tensor_C(previous_hidden_num =796,next_hidden_num=8,classes=8):
    tensor_C = np.zeros(previous_hidden_num*next_hidden_num).reshape(previous_hidden_num,next_hidden_num)

    top_left_nums = int(math.floor(previous_hidden_num / classes))
    column =top_left_nums
    row =  int(math.floor(next_hidden_num /classes))
    top_left = [[i * column, i * row] for i in range(classes)]

    remainder_1 = previous_hidden_num % classes
    remainder_2 = next_hidden_num % classes

    base_matrix = []
    for i in range(column):
        for j in range(row):
            base_matrix.append([i,j])
    base_matrix_1 = np.array(base_matrix)

    base_matrix = []
    for i in range(column+remainder_1):
        for j in range(row+remainder_2):
            base_matrix.append([i,j])
    base_matrix_2 = np.array(base_matrix)


    matrix_one_1 = [(base_matrix_1 + i).tolist() for i in top_left[:-1]]

    matrix_one_1_1 = []    
    for item in matrix_one_1:
        matrix_one_1_1 = matrix_one_1_1 + item

    matrix_one_2 = (base_matrix_2 + top_left[-1]).tolist()
    matrix_one = matrix_one_1_1 + matrix_one_2

    for item in range(len(matrix_one)):
        tensor_C[matrix_one[item][0],matrix_one[item][1]] = 1
    
    tensor_C = Variable(torch.from_numpy(tensor_C.astype("float32")).cuda())
    return tensor_C



class OS_Linear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(OS_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.dim() == 2 and self.bias is not None:
            tensor_C = design_tensor_C(classes=self.out_features)
            output = input.matmul(self.weight.t()* tensor_C)

            if self.bias is not None:
                output += self.bias

            return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'