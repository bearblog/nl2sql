import torch
from torch import nn
from torch.nn import functional as F
class TableAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TableAttention, self).__init__()
        self.linear_layer = nn.Linear(input_size, hidden_size, bias=False)

    def mask_and_acv(self, output, output_mask):
        # 对 output mask，然后求 Relu(WX)
        batch_size, output_len, hidden_size = output.shape
        output_mask = output_mask.unsqueeze(2).repeat(1, 1, hidden_size).view(batch_size, output_len, hidden_size)
        output = torch.mul(output, output_mask)
        output = F.relu(self.linear_layer(output))
        return output

    def forward(self, seq_output, seq_mask, target_output, target_mask):
        batch_size, seq_len, hidden_size = seq_output.shape
        batch_size, target_len, hidden_size = target_output.shape
        seq_mask = seq_mask.float()
        target_mask = target_mask.float()
        # 对 seq_output 和 target_output 都 mask，并 Relu
        seq_output_transformed = self.mask_and_acv(seq_output, seq_mask)
        target_output_transformed = self.mask_and_acv(target_output, target_mask)
        # seq_output_transformed 和 target_output_transformed 每一列两两点乘（处理成矩阵相乘），最后一个维度扩展成 hidden_size，最后是 (batch_size, seq_len, target_len, hidden_size)
        attention_matrix = torch.matmul(seq_output_transformed, target_output_transformed.transpose(2, 1))
        attention_matrix = F.softmax(attention_matrix.float(), dim=2)
        attention_matrix_unsqueeze = attention_matrix.unsqueeze(3).repeat(1, 1, 1, hidden_size)
        # 将 target_output 第二个维度 repeat seq_len，(batch_size, target_len, hidden_size) -> (batch_size, seq_len, target_len, hidden_size)，和注意力矩阵相乘后对第三个维度求sum
        target_output_unsqueeze = target_output.unsqueeze(1).repeat(1, seq_len, 1, 1)
        attention_output_unsqueeze = torch.mul(target_output_unsqueeze, attention_matrix_unsqueeze)
        attention_output = torch.sum(attention_output_unsqueeze, 2)
        # attention_output 拼接 seq_output，再用 seq_mask mask 一遍
        cat_output = torch.cat([seq_output, attention_output], 2)
        cat_mask = seq_mask.unsqueeze(2).repeat(1, 1, hidden_size * 2).view(batch_size, seq_len, hidden_size * 2)
        cat_output = torch.mul(cat_output, cat_mask)
        return cat_output, attention_output
