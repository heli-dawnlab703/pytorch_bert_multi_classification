from transformers import BertModel
import torch.nn as nn
import torch

class BertForMultiLabelClassification(nn.Module):
    def __init__(self, args):
        super(BertForMultiLabelClassification, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.bert_config = self.bert.config
        out_dims = self.bert_config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(out_dims, args.num_tags)

    def forward(self, token_ids, attention_masks, token_type_ids):
        bert_outputs = self.bert(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
        )
        seq_out = bert_outputs[1]
        seq_out = self.dropout(seq_out)
        seq_out = self.linear(seq_out)
        return seq_out


class BertForRank(nn.Module):
    def __init__(self, args):
        super(BertForRank, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.bert_config = self.bert.config
        out_dims = self.bert_config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(out_dims, args.num_tags)
        # 初始化标签嵌入
        self.label_num = args.num_tags
        self.tanh = nn.Tanh()

        # self.label = nn.Parameter(torch.randn(self.label_num, 768))

        self.label = torch.zeros((6, 768)).to(self.device)
        self.label[torch.arange(6), torch.arange(6)] = 1

    def forward(self, token_ids, attention_masks, token_type_ids):
        bert_outputs = self.bert(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
        )
        seq_out = bert_outputs[1]  # (batch,dim)
        seq_out = self.dropout(seq_out)
        batch_size = seq_out.shape[0]
        dim = seq_out.shape[1]
        label = self.label.unsqueeze(0).repeat(batch_size,1,1) # (batch,label_num,dim)
        seq_out = seq_out.unsqueeze(1).repeat(1,self.label_num,1) # (batch,label_num,dim)

        # 以tanh激活
        # seq_out = self.tanh(seq_out)
        # label = self.tanh(label)
        
        # 计算内积
        relate_score = torch.einsum("ijk,ijk->ij",seq_out,label) / dim
        return relate_score
