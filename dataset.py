import sys
sys.path.append('..')
import torch
from torch.utils.data import Dataset, DataLoader
# 这里要显示的引入BertFeature，不然会报错
from preprocess import BertFeature
from preprocess import get_out, Processor
import bert_config


class MLDataset(Dataset):
    def __init__(self, features):
        self.nums = len(features)

        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks).float() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]
        self.labels = [torch.tensor(example.labels).float() for example in features]

        # 计算标签权重
        lab = [example.labels for example in features]
        lab_dict = {}
        for i in lab:
            idx = i.index(1)
            ct = lab_dict.get(idx,0)
            lab_dict[idx] = ct + 1
        all_ct = 0
        for i in lab_dict:
            all_ct += lab_dict[i]
        for i in lab_dict:
            lab_dict[i] = lab_dict[i] / all_ct
        for i in lab_dict:
            lab_dict[i] = 1 / lab_dict[i]
        self.weight = [lab_dict[example.labels.index(1)] for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {
            'token_ids': self.token_ids[index],
            'attention_masks': self.attention_masks[index],
            'token_type_ids': self.token_type_ids[index]
        }

        data['labels'] = self.labels[index]

        return data

if __name__ == '__main__':
    args = bert_config.Args().get_parser()
    args.log_dir = './logs/'
    args.max_seq_len = 128
    args.bert_dir = '../model_hub/bert-base-chinese/'

    processor = Processor()

    label2id = {}
    id2label = {}
    with open('./data/final_data/labels.txt', 'r') as fp:
        labels = fp.read().strip().split('\n')
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[id] = label
    print(label2id)

    train_out = get_out(processor, './data/raw_data/train.json', args, label2id, 'train')
    features, callback_info = train_out
    train_dataset = MLDataset(features)
    for data in train_dataset:
        print(data['token_ids'])
        print(data['attention_masks'])
        print(data['token_type_ids'])
        print(data['labels'])
        break