import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
from modules.value_attention import ValueAttention
from modules.table_attention import TableAttention
# from pytorch_pretrained_bert import BertTokenizer, BertAdam, BertModel
# from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from pytorch_transformers import *
from pytorch_transformers.modeling_bert import BertPreTrainedModel


class BertNL2SQL(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNL2SQL, self).__init__(config)  # BertConfig
        self.num_tag_labels = 2
        self.num_agg_labels = 6
        self.num_connection_labels = 3
        self.num_con_num_labels = 4
        self.num_type_labels = 3
        self.num_sel_num_labels = 4  # {0, 1, 2, 3}
        self.num_where_num_labels = 5  # {0, 1, 2, 3, 4}
        self.num_op_labels = 4
        self.hidden_size = config.hidden_size
        config.output_hidden_states = True
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear_tag = nn.Linear(self.hidden_size * 3, self.num_tag_labels)
        self.linear_agg = nn.Linear(self.hidden_size * 2, self.num_agg_labels)
        self.linear_connection = nn.Linear(self.hidden_size, self.num_connection_labels)
        self.linear_con_num = nn.Linear(self.hidden_size * 2, self.num_con_num_labels)
        self.linear_type = nn.Linear(self.hidden_size * 2, self.num_type_labels)
        self.linear_sel_num = nn.Linear(self.hidden_size, self.num_sel_num_labels)
        self.linear_where_num = nn.Linear(self.hidden_size, self.num_where_num_labels)
        self.values_attention = TableAttention(self.hidden_size, self.hidden_size)
        self.head_attention = ValueAttention(self.hidden_size, self.hidden_size)
        self.linear_op = nn.Linear(self.hidden_size * 2, self.num_op_labels)
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask, all_masks, header_masks, question_masks, subheader_masks,
                nextColumn_CLS_startPositionList, value_masks, firstColumn_CLS_startPositionList,
                train_dependencies=None):
        outputs = self.bert(input_ids, None, attention_mask)
        # The last hidden-state is the first element of the output tuple
        sequence_output = outputs[0]
        # sequence_output = outputs[2][-2]
        device = "cuda" if torch.cuda.is_available() else None
        type_masks = all_masks.view(-1) == 1
        # TODO: 如果是求平均值就计算每行mask总和，mask，每行相加除以每行总和
        cls_output = sequence_output[type_masks, firstColumn_CLS_startPositionList[type_masks], :]
        _, subheader_attention = self.head_attention(cls_output, sequence_output, subheader_masks)
        cat_cls = torch.cat([cls_output, subheader_attention], 1)

        cat_output, _ = self.values_attention(sequence_output, question_masks, sequence_output, value_masks)
        _, header_attention = self.values_attention(sequence_output, question_masks, sequence_output, header_masks)
        cat_output = torch.cat([cat_output, header_attention], 2)

        num_output = sequence_output[type_masks, 0, :]
        if train_dependencies:
            tag_masks = train_dependencies[0].view(-1) == 1
            sel_masks = train_dependencies[1].view(-1) == 1
            con_masks = train_dependencies[2].view(-1) == 1
            type_masks = all_masks.view(-1) == 1
            connection_labels = train_dependencies[3]
            agg_labels = train_dependencies[4]
            tag_labels = train_dependencies[5]
            con_num_labels = train_dependencies[6]
            type_labels = train_dependencies[7]
            sel_num_labels = train_dependencies[8]
            where_num_labels = train_dependencies[9]
            op_labels = train_dependencies[10]
            # mask 后的 bert_output
            tag_output = cat_output.contiguous().view(-1, self.hidden_size * 3)[tag_masks]
            tag_labels = tag_labels.view(-1)[tag_masks]
            agg_output = cat_cls[sel_masks, :]
            agg_labels = agg_labels[sel_masks]
            connection_output = sequence_output[con_masks, 0, :]
            connection_labels = connection_labels[con_masks]
            con_num_output = cat_cls[con_masks, :]
            con_num_labels = con_num_labels[con_masks]
            op_output = cat_cls[con_masks, :]
            op_labels = op_labels[con_masks]
            type_output = cat_cls[type_masks, :]
            type_labels = type_labels[type_masks]
            # 全连接层
            tag_output = self.linear_tag(self.dropout(tag_output))
            agg_output = self.linear_agg(self.dropout(agg_output))
            connection_output = self.linear_connection(self.dropout(connection_output))
            con_num_output = self.linear_con_num(self.dropout(con_num_output))
            type_output = self.linear_type(self.dropout(type_output))
            sel_num_output = self.linear_sel_num(self.dropout(num_output))
            where_num_output = self.linear_where_num(self.dropout(num_output))
            op_output = self.linear_op(self.dropout(op_output))
            # 损失函数
            loss_function = nn.CrossEntropyLoss(reduction="mean")
            tag_loss = loss_function(tag_output, tag_labels)
            agg_loss = loss_function(agg_output, agg_labels)
            connection_loss = loss_function(connection_output, connection_labels)
            con_num_loss = loss_function(con_num_output, con_num_labels)
            type_loss = loss_function(type_output, type_labels)
            sel_num_loss = loss_function(sel_num_output, sel_num_labels)
            where_num_loss = loss_function(where_num_output, where_num_labels)
            op_loss = loss_function(op_output, op_labels)
            loss = tag_loss + agg_loss + connection_loss + con_num_loss + type_loss + sel_num_loss + where_num_loss + op_loss
            return loss
        else:
            # 用于evaluate 和 预测
            all_masks = all_masks.view(-1) == 1
            batch_size, seq_len, hidden_size = sequence_output.shape
            tag_output = torch.zeros(batch_size, seq_len, hidden_size * 3, dtype=torch.float32, device=device)
            for i in range(batch_size):
                for j in range(seq_len):
                    if attention_mask[i][j] == 1:
                        tag_output[i][j] = cat_output[i][j]
            head_output = sequence_output[:, 0, :]
            # cls_output = sequence_output[all_masks, firstColumn_CLS_startPositionList, :]
            tag_output = self.linear_tag(self.dropout(tag_output))
            agg_output = self.linear_agg(self.dropout(cat_cls))
            connection_output = self.linear_connection(self.dropout(head_output))
            con_num_output = self.linear_con_num(self.dropout(cat_cls))
            type_output = self.linear_type(self.dropout(cat_cls))
            sel_num_output = self.linear_sel_num(self.dropout(num_output))
            where_num_output = self.linear_where_num(self.dropout(num_output))
            op_output = self.linear_op(self.dropout(cat_cls))

            tag_probs = F.log_softmax(tag_output, dim=2).detach().cpu().numpy().tolist()
            agg_probs = F.log_softmax(agg_output, dim=1).detach().cpu().numpy().tolist()
            connection_probs = F.log_softmax(connection_output, dim=1).detach().cpu().numpy().tolist()
            con_num_probs = F.log_softmax(con_num_output, dim=1).detach().cpu().numpy().tolist()
            type_probs = F.log_softmax(type_output, dim=1).detach().cpu().numpy().tolist()
            sel_num_probs = F.log_softmax(sel_num_output, dim=1).detach().cpu().numpy().tolist()
            where_num_probs = F.log_softmax(where_num_output, dim=1).detach().cpu().numpy().tolist()
            op_probs = F.log_softmax(op_output, dim=1).detach().cpu().numpy().tolist()

            tag_logits = torch.argmax(F.log_softmax(tag_output, dim=2), dim=2).detach().cpu().numpy().tolist()
            agg_logits = torch.argmax(F.log_softmax(agg_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            connection_logits = torch.argmax(F.log_softmax(connection_output, dim=1),
                                             dim=1).detach().cpu().numpy().tolist()
            con_num_logits = torch.argmax(F.log_softmax(con_num_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            type_logits = torch.argmax(F.log_softmax(type_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            sel_num_logits = torch.argmax(F.log_softmax(sel_num_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            where_num_logits = torch.argmax(F.log_softmax(where_num_output, dim=1),
                                            dim=1).detach().cpu().numpy().tolist()
            op_logits = torch.argmax(F.log_softmax(op_output, dim=1), dim=1).detach().cpu().numpy().tolist()

            return tag_logits, agg_logits, connection_logits, con_num_logits, type_logits, sel_num_logits, where_num_logits, type_probs, op_logits
