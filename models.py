import os

from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch


def gaussian_jsd(g1, g2):
    """
    计算两个高斯分布序列的JS散度
    :return:
    """
    mean1, conv1 = g1
    mean2, conv2 = g2

    dist1 = torch.distributions.MultivariateNormal(mean1, conv1)
    dist2 = torch.distributions.MultivariateNormal(mean2, conv2)

    kl_div_12 = torch.distributions.kl_divergence(dist1, dist2)
    kl_div_21 = torch.distributions.kl_divergence(dist2, dist1)

    js_div = 0.5 * (kl_div_12 + kl_div_21)
    return js_div


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

        # gru相关
        self.word_emb = nn.Embedding(40000, embedding_dim=out_dims)
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

        cls_token = bert_outputs.last_hidden_state[:, 0, :]  # 句子级别
        cls_token = cls_token.unsqueeze(1)  # (batch,1,dim)
        bert_token_enc = bert_outputs.last_hidden_state

        # gru
        gru_inp = self.word_emb(token_ids)
        gru_output, gru_hn = self.bigru(gru_inp)
        gru_output = self.linear_gru(gru_output)
        gru_hn = self.linear_gru1(gru_hn.transpose(0, 2)).transpose(0, 2).transpose(0, 1)

        # bert_token_enc(batch,seq,dim)
        # cls_token(batch,1,dim)
        # gru_output(batch,seq,dim)
        # gru_hn(batch,1,dim)
        return (bert_token_enc + gru_output) / 2, (cls_token + gru_hn) / 2


class Gaussian_attn(nn.Module):
    def __init__(self, args):
        super(Gaussian_attn, self).__init__()
        self.softmax = nn.Softmax(-1)

    def attn_score(self, q, k,mask):
        q_mean, q_div = q  # (batch,label_num,dim)  (batch,label_num,dim,dim)
        k_mean, k_div = k  # (batch,seq,dim)  (batch,seq,dim,dim)

        q_seq = q_mean.shape[1]
        k_seq = k_mean.shape[1]
        batch = q_mean.shape[0]
        dim = q_mean.shape[2]

        q_mean = q_mean.unsqueeze(2).repeat(1, 1, k_seq, 1).view(batch, -1, dim)
        k_mean = k_mean.unsqueeze(1).repeat(1, q_seq, 1, 1).view(batch, -1, dim)

        q_div = q_div.unsqueeze(2).repeat(1, 1, k_seq, 1, 1).view(batch, -1, dim, dim)
        k_div = k_div.unsqueeze(1).repeat(1, q_seq, 1, 1, 1).view(batch, -1, dim, dim)

        score = gaussian_jsd((q_mean,q_div),(k_mean,k_div)).view(batch,q_seq,k_seq) # (batch,q_seq * k_seq)
        if mask:
            score = score + mask
        score = self.softmax(score)

        return score

    def forward(self, q, k, v, mask=None):
        # q_mean, q_div = q  # (batch,label_num,dim)  (batch,label_num,dim,dim)
        # k_mean, k_div = k  # (batch,seq,dim)  (batch,seq,dim,dim)
        # v_mean, v_div = v  # (batch,seq,dim)  (batch,seq,dim,dim)
        # q_len = q_mean.shape[1]

        score = self.attn_score(q,k,mask)
        v_mean = torch.einsum("bqk,bkd->bqd",score,v[0])
        v_div = torch.einsum("bqk,bkdm->bqdm",score,v[1])

        return v_mean,v_div


class BertForRank(nn.Module):
    def __init__(self, args):
        super(BertForRank, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dp = 0.1
        self.dropout = nn.Dropout(dp)
        # 初始化标签嵌入
        self.label_num = args.num_tags
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
        self.init_label_vector(args, "rand")
        self.mean_encoder = TextEncoder(args)
        self.div_encoder = TextEncoder(args)
        self.gs_attn = Gaussian_attn(args)
        out_dims = self.mean_encoder.bert_config.hidden_size
        self.jw1 = nn.Linear(out_dims,128)
        self.jw2 = nn.Linear(out_dims,128)
        self.jw3 = nn.Linear(out_dims,128)
        self.jw4 = nn.Linear(out_dims,128)

    def init_label_vector(self, args, choose):
        if choose == "rand":
            # self.label_mean = nn.Parameter(torch.randn(self.label_num, args.bert_label_hidden))
            # self.label_div = nn.Parameter(torch.randn(self.label_num, args.bert_label_hidden))
            # 临时做个降维处理
            self.label_mean = nn.Parameter(torch.randn(self.label_num, 128))
            self.label_div = nn.Parameter(torch.randn(self.label_num, 128))

    def forward(self, token_ids, attention_masks, token_type_ids):
        # text的encode
        # word (batch,seq,dim)
        # sent (batch,1,dim)
        word_mean, sent_mean = self.mean_encoder(token_ids,
                                                 attention_masks,
                                                 token_type_ids)
        word_div, sent_div = self.div_encoder(token_ids, attention_masks,
                                              token_type_ids)

        word_mean = self.jw1(word_mean)
        word_div = self.jw2(word_div)
        sent_mean = self.jw3(sent_mean)
        sent_div = self.jw4(sent_div)

        # 获取batch和dim
        batch_size = word_mean.shape[0]
        dim = word_mean.shape[-1]
        seq_num = word_mean.shape[1]

        word_div = word_div.unsqueeze(-1) * word_div.unsqueeze(-2) + torch.eye(dim).unsqueeze(0).unsqueeze(0).repeat(batch_size,seq_num,1,1).to(self.device)
        sent_div = sent_div.unsqueeze(-1) * sent_div.unsqueeze(-2) + torch.eye(dim).unsqueeze(0).unsqueeze(0).repeat(batch_size,1,1,1).to(self.device)

        # label的encode
        # label_mean (batch,label_num,dim)
        # label_div (batch,label_num,dim,dim)
        label_mean = self.label_mean.unsqueeze(0).repeat(batch_size, 1, 1)
        label_div = self.label_div.unsqueeze(-1) * self.label_div.unsqueeze(-2) + torch.eye(dim).unsqueeze(0).expand(self.label_num, -1, -1).to(self.device)
        label_div = label_div.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # label聚合单词
        word_mean,word_div = self.gs_attn((label_mean,label_div),(word_mean,word_div),(word_mean,word_div))
        sent_mean = sent_mean.repeat(1,self.label_num,1)
        sent_div = sent_div.repeat(1,self.label_num,1,1)
        text_mean = (sent_mean + word_mean) / 2
        text_div = (sent_div + word_div) / 2

        # 计算内积 (batch,lab_num)
        relate_score = gaussian_jsd((text_mean,text_div),(label_mean,label_div))

        return relate_score, label_mean, label_div