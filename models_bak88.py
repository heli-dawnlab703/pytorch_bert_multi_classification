import os

from transformers import BertModel, BertTokenizer
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


class TextEncoder(nn.Module):
    def __init__(self, args):
        super(TextEncoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_llm(args=args, choose="unfixed")
        self.bert_config = self.bert.config
        out_dims = self.bert_config.hidden_size
        dp = 0.1
        self.dropout = nn.Dropout(dp)
        # 初始化标签嵌入
        self.label_num = args.num_tags
        self.init_label_vector(args, "rand")
        # 注意力层
        self.attn_llm = nn.MultiheadAttention(embed_dim=out_dims, num_heads=2, dropout=dp,
                                              batch_first=True)  # 以目标问题聚合历史问题
        self.attn_gru = nn.MultiheadAttention(embed_dim=out_dims, num_heads=2, dropout=dp,
                                              batch_first=True)  # 以目标问题聚合历史问题
        self.word_emb = nn.Embedding(50000, embedding_dim=out_dims)

        gru_layer = 2
        self.bigru = nn.GRU(input_size=out_dims, hidden_size=out_dims, num_layers=gru_layer, batch_first=True,
                            dropout=dp, bidirectional=True)
        self.linear_gru = nn.Linear(2 * out_dims, out_dims)
        self.linear_gru1 = nn.Linear(gru_layer * 2, 1)

    def init_llm(self, args, choose):
        self.bert = BertModel.from_pretrained(args.bert_dir)
        for param in self.bert.parameters():
            if choose == "unfixed":
                param.requires_grad = True  # 是否进行Finetune
            else:
                param.requires_grad = False


    def forward(self, token_ids, attention_masks, token_type_ids):
        bert_outputs = self.bert(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
        )

        batch_size = bert_outputs.pooler_output.shape[0]

        # 以注意力进行聚合
        seq_out, _ = self.attn_llm(self.label.unsqueeze(0).repeat(batch_size, 1, 1), bert_outputs.last_hidden_state,
                                   bert_outputs.last_hidden_state, key_padding_mask=attention_masks.float())
        cls_token = bert_outputs.last_hidden_state[:, 0, :]  # 句子级别
        cls_token = cls_token.unsqueeze(1).repeat(1, self.label_num, 1)  # (batch,label_num,dim)

        # gru
        gru_inp = self.word_emb(token_ids)
        gru_output, gru_hn = self.bigru(gru_inp)
        gru_output = self.linear_gru(gru_output)
        gru_output, _ = self.attn_gru(self.label.unsqueeze(0).repeat(batch_size, 1, 1), gru_output, gru_output,
                                      key_padding_mask=attention_masks.float())
        gru_hn = self.linear_gru1(gru_hn.transpose(0, 2)).transpose(0, 2).transpose(0, 1)
        gru_hid = gru_hn.repeat(1, self.label_num, 1)

        seq_out = (self.dropout(seq_out) + self.dropout(cls_token) + self.dropout(gru_hid) + self.dropout(
            gru_output)) / 4

        return seq_out


class BertForRank(nn.Module):
    def __init__(self, args):
        super(BertForRank, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_llm(args=args, choose="unfixed")
        self.bert_config = self.bert.config
        out_dims = self.bert_config.hidden_size
        dp = 0.1
        self.dropout = nn.Dropout(dp)
        self.linear = nn.Linear(out_dims, args.num_tags)
        # 初始化标签嵌入
        self.label_num = args.num_tags
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
        self.init_label_vector(args, "rand")

        # 注意力层
        self.attn_llm = nn.MultiheadAttention(embed_dim=out_dims, num_heads=2, dropout=dp,
                                              batch_first=True)  # 以目标问题聚合历史问题
        self.attn_gru = nn.MultiheadAttention(embed_dim=out_dims, num_heads=2, dropout=dp,
                                              batch_first=True)  # 以目标问题聚合历史问题
        self.word_emb = nn.Embedding(50000, embedding_dim=out_dims)

        gru_layer = 2
        self.bigru = nn.GRU(input_size=out_dims, hidden_size=out_dims, num_layers=gru_layer, batch_first=True,
                            dropout=dp, bidirectional=True)
        self.linear_gru = nn.Linear(2 * out_dims, out_dims)
        self.linear_gru1 = nn.Linear(gru_layer * 2, 1)

    def init_llm(self, args, choose):
        self.bert = BertModel.from_pretrained(args.bert_dir)
        for param in self.bert.parameters():
            if choose == "unfixed":
                param.requires_grad = True  # 是否进行Finetune
            else:
                param.requires_grad = False

    def init_label_vector(self, args, choose):
        if choose == "rand":
            self.label = nn.Parameter(torch.randn(self.label_num, args.bert_label_hidden))
        elif choose == "bert":
            tokenizer = BertTokenizer(os.path.join(args.bert_dir, 'vocab.txt'))
            with open("./data/final_data/class_chinese.txt") as f:
                label = f.readlines()
                label = [i.split()[0] for i in label]
                label_vector = []
                for lab in label:
                    encode_dict = tokenizer.encode_plus(text=lab,
                                                        add_special_tokens=False,
                                                        return_attention_mask=False,
                                                        return_tensors='pt'
                                                        )
                    bert_outputs = self.bert(encode_dict.data["input_ids"]).pooler_output
                    label_vector.append(bert_outputs)
                self.label = torch.nn.Parameter(torch.concat(label_vector, dim=0).to(self.device).detach())

    def forward(self, token_ids, attention_masks, token_type_ids):
        bert_outputs = self.bert(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
        )

        batch_size = bert_outputs.pooler_output.shape[0]
        dim = bert_outputs.pooler_output.shape[1]
        label = self.label.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch,label_num,dim)

        # 以注意力进行聚合
        seq_out, _ = self.attn_llm(self.label.unsqueeze(0).repeat(batch_size, 1, 1), bert_outputs.last_hidden_state,
                                   bert_outputs.last_hidden_state, key_padding_mask=attention_masks.float())
        cls_token = bert_outputs.last_hidden_state[:, 0, :]  # 句子级别
        cls_token = cls_token.unsqueeze(1).repeat(1, self.label_num, 1)  # (batch,label_num,dim)

        # gru
        gru_inp = self.word_emb(token_ids)
        gru_output, gru_hn = self.bigru(gru_inp)
        gru_output = self.linear_gru(gru_output)
        gru_output, _ = self.attn_gru(self.label.unsqueeze(0).repeat(batch_size, 1, 1), gru_output, gru_output,
                                      key_padding_mask=attention_masks.float())
        gru_hn = self.linear_gru1(gru_hn.transpose(0, 2)).transpose(0, 2).transpose(0, 1)
        gru_hid = gru_hn.repeat(1, self.label_num, 1)

        # seq_out = (seq_out + cls_token + gru_hid + gru_output) / 4
        seq_out = (self.dropout(seq_out) + self.dropout(cls_token) + self.dropout(gru_hid) + self.dropout(
            gru_output)) / 4

        # 以tanh激活
        seq_out = self.tanh(seq_out)
        label = self.tanh(label)

        # 计算内积
        relate_score = torch.einsum("ijk,ijk->ij", seq_out, label) / dim / 2
        return relate_score, 0, label
