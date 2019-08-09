import torch
from torch import nn
from torch.nn import functional as F
class ValueAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ValueAttention, self).__init__()
        self.linear_layer = nn.Linear(input_size, hidden_size, bias=False)

    def mask_and_acv(self, output, output_mask):
        # 对 output mask，然后求 Relu(WX)
        batch_size, output_len, hidden_size = output.shape
        output_mask = output_mask.unsqueeze(2).repeat(1, 1, hidden_size).view(batch_size, output_len, hidden_size)
        output = torch.mul(output, output_mask)
        output = F.relu(self.linear_layer(output))
        return output

    def forward(self, col_output, target_output, target_mask):
        batch_size, target_len, hidden_size = target_output.shape
        target_mask = target_mask.float()
        col_output_transformed = F.relu(self.linear_layer(col_output))
        target_output_transformed = self.mask_and_acv(target_output, target_mask)
        attention_matrix = torch.matmul(col_output_transformed.unsqueeze(1), target_output_transformed.transpose(2, 1)).squeeze(1)
        attention_matrix = F.softmax(attention_matrix.float(), dim=1)
        attention_matrix_unsqueeze = attention_matrix.unsqueeze(2).repeat(1, 1, hidden_size)
        attention_output_unsqueeze = torch.mul(target_output, attention_matrix_unsqueeze)
        attention_output = torch.sum(attention_output_unsqueeze, 1)
        cat_output = torch.cat([col_output, attention_output], 1)
        return cat_output, attention_output

