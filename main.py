import re
from pprint import pprint
import os
import logging
import shutil
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix, classification_report
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer
import bert_config
import preprocess
import dataset
import models
import utils

logger = logging.getLogger(__name__)

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
os.environ["CUDA_HOME"] = "/usr/local/cuda"

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class MarginLoss(nn.Module):
    def __init__(self,margin):
        super().__init__()
        self.margin = margin
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, score, label): # score(batch,label_num)  label(batch,label_num)
        batch = score.shape[0]
        # print(score.shape)
        pos = torch.masked_select(score, label>0.5).view(batch,-1)
        # print(pos)
        neg = torch.masked_select(score, label<0.5).view(batch,-1)
        # print(neg.shape)
        neg_num = neg.shape[1]
        pos = pos.repeat(1,neg_num)
        zero_tensor = torch.zeros(batch, neg_num).to(self.device)
        loss = torch.max(-pos + neg + self.margin, zero_tensor)
        loss = torch.sum(loss) / neg_num
        return loss


class Trainer:
    def __init__(self, args, train_loader, dev_loader, test_loader):
        self.args = args
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.model_name = args.model_name

        if args.model_name == "classification":
            self.model = models.BertForMultiLabelClassification(args)
            self.criterion = nn.BCEWithLogitsLoss()
        elif args.model_name == "rank":
            self.model = models.BertForRank(args)
            self.criterion = MarginLoss(margin=1)
        else:
            raise "unknown model"
        print(get_parameter_number(self.model))
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.model.to(self.device)

    def margin_loss(self, score, label):
        pass

    def load_ckp(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)

    """
    def save_ckp(self, state, is_best, checkpoint_path, best_model_path):
        tmp_checkpoint_path = checkpoint_path
        torch.save(state, tmp_checkpoint_path)
        if is_best:
            tmp_best_model_path = best_model_path
            shutil.copyfile(tmp_checkpoint_path, tmp_best_model_path)
    """

    def train(self):
        total_step = len(self.train_loader) * self.args.train_epochs
        global_step = 0
        eval_step = 100
        best_dev_micro_f1 = 0.0
        for epoch in range(args.train_epochs):
            step_count = 0
            epo_mean_loss = 0.
            for train_step, train_data in enumerate(self.train_loader):
                self.model.train()
                token_ids = train_data['token_ids'].to(self.device)
                attention_masks = train_data['attention_masks'].to(self.device)
                token_type_ids = train_data['token_type_ids'].to(self.device)
                labels = train_data['labels'].to(self.device)
                train_outputs = self.model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(train_outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epo_mean_loss += loss.item()
                step_count += 1
                logger.info(
                    "【train】 epoch：{} step:{}/{} loss：{:.4f} mean_loss：{:.4f}".format(epoch, global_step, total_step, loss.item(),epo_mean_loss/step_count))
                global_step += 1
                if global_step % eval_step == 0:
                    dev_loss, dev_outputs, dev_targets = self.dev()
                    accuracy, micro_f1, macro_f1 = self.get_metrics(dev_outputs, dev_targets)
                    logger.info(
                        "【dev】 loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(dev_loss, accuracy,
                                                                                                   micro_f1, macro_f1))
                    if macro_f1 > best_dev_micro_f1:
                        logger.info("------------>保存当前最好的模型")
                        checkpoint = {
                            'epoch': epoch,
                            'loss': dev_loss,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }
                        best_dev_micro_f1 = macro_f1
                        checkpoint_path = os.path.join(self.args.output_dir, 'best.pt')
                        self.save_ckp(checkpoint, checkpoint_path)

    def dev(self):
        self.model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(self.dev_loader):
                token_ids = dev_data['token_ids'].to(self.device)
                attention_masks = dev_data['attention_masks'].to(self.device)
                token_type_ids = dev_data['token_type_ids'].to(self.device)
                labels = dev_data['labels'].to(self.device)
                outputs = self.model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(outputs, labels)
                # val_loss = val_loss + ((1 / (dev_step + 1))) * (loss.item() - val_loss)
                total_loss += loss.item()
                if self.model_name == "classification":
                    outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                    outputs = (np.array(outputs) > 0.6).astype(int)
                else:
                    outputs = np.array(outputs.cpu().detach().numpy().tolist())
                    tmp = np.zeros_like(outputs)
                    tmp[np.arange(len(outputs)), outputs.argmax(1)] = 1
                    outputs = tmp
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, dev_outputs, dev_targets

    def test(self, checkpoint_path):
        model = self.model
        optimizer = self.optimizer
        model, optimizer, epoch, loss = self.load_ckp(model, optimizer, checkpoint_path)
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_step, test_data in enumerate(self.test_loader):
                token_ids = test_data['token_ids'].to(self.device)
                attention_masks = test_data['attention_masks'].to(self.device)
                token_type_ids = test_data['token_type_ids'].to(self.device)
                labels = test_data['labels'].to(self.device)
                outputs = model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(outputs, labels)
                # val_loss = val_loss + ((1 / (dev_step + 1))) * (loss.item() - val_loss)
                total_loss += loss.item()
                if self.model_name == "classification":
                    outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                    outputs = (np.array(outputs) > 0.6).astype(int)
                else:
                    outputs = np.array(outputs.cpu().detach().numpy().tolist())
                    tmp = np.zeros_like(outputs)
                    tmp[np.arange(len(outputs)), outputs.argmax(1)] = 1
                    outputs = tmp
                test_outputs.extend(outputs.tolist())
                test_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, test_outputs, test_targets

    def predict(self, tokenizer, text, id2label, args):
        model = self.model
        optimizer = self.optimizer
        checkpoint = os.path.join(args.output_dir, 'best.pt')
        model, optimizer, epoch, loss = self.load_ckp(model, optimizer, checkpoint)
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            inputs = tokenizer.encode_plus(text=text,
                                           add_special_tokens=True,
                                           max_length=args.max_seq_len,
                                           truncation='longest_first',
                                           padding="max_length",
                                           return_token_type_ids=True,
                                           return_attention_mask=True,
                                           return_tensors='pt')
            token_ids = inputs['input_ids'].to(self.device)
            attention_masks = inputs['attention_mask'].to(self.device)
            token_type_ids = inputs['token_type_ids'].to(self.device)
            outputs = model(token_ids, attention_masks, token_type_ids)
            
            if self.model_name == "classification":
                outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                outputs = (np.array(outputs) > 0.5).astype(int)
                outputs = np.where(outputs[0] == 1)[0].tolist()
            else:
                outputs = np.array(outputs.cpu().detach().numpy().tolist())
                tmp = np.zeros_like(outputs)
                tmp[np.arange(len(outputs)), outputs.argmax(1)] = 1
                outputs = tmp[0].tolist()

            if len(outputs) != 0:
                if self.model == 'classification':
                    outputs = [id2label[i] for i in outputs]
                else:
                    indices = [i for i, x in enumerate(outputs) if x == 1]
                    # outputs = [id2label[i] for i in indices]
                    outputs = indices
                return outputs

            else:
                return '不好意思，我没有识别出来'

    def get_metrics(self, outputs, targets):
        accuracy = accuracy_score(targets, outputs)
        micro_f1 = f1_score(targets, outputs, average='micro')
        macro_f1 = f1_score(targets, outputs, average='macro')
        return accuracy, micro_f1, macro_f1

    def get_classification_report(self, outputs, targets, labels):
        # confusion_matrix = multilabel_confusion_matrix(targets, outputs)
        report = classification_report(targets, outputs, target_names=labels)
        return report


def convert_json_label():
    global fp, labels
    with open('./data/final_data/class.txt', 'r', encoding='utf-8') as fp:
        labels = fp.read().strip().split('\n')
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label


def convert_txt_label():
    global fp, labels
    with open('./data/final_data/class.txt', 'r', encoding='utf-8') as fp:
        labels = fp.read().strip().split('\n')
    for i, label in enumerate(labels):
        split_result = re.split(r'\s+', label)
        label2id[split_result[0]] = i
        id2label[i] = split_result[0]


def predict_json():
    global fp, text
    # 读取test1.json里面的数据
    with open(os.path.join('./data/raw_data/test1.json'), 'r') as fp:
        lines = fp.read().strip().split('\n')
        for line in lines:
            text = eval(line)['text']
            print(text)
            result = trainer.predict(tokenizer, text, id2label, args)
            print(result)


def predict_txt():
    global fp, text
    # 读取test1.json里面的数据
    with open(os.path.join('./data/tianchi/dev.txt'), 'r') as fp:
        lines = fp.read().strip().split('\n')
        for line in lines:
            split_result = re.split(r'\t', line)
            text = split_result[0]
            print(text)
            result = trainer.predict(tokenizer, text, id2label, args)
            print(result)


if __name__ == '__main__':
    args = bert_config.Args().get_parser()
    utils.utils.set_seed(args.seed)
    utils.utils.set_logger(os.path.join(args.log_dir, 'main.log'))

    processor = preprocess.Processor()

    label2id = {}
    id2label = {}
    convert_txt_label()
    print(label2id)

    train_out = preprocess.get_out(processor, './data/tianchi/train.txt', args, label2id, 'train')
    train_features, train_callback_info = train_out
    train_dataset = dataset.MLDataset(train_features)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size,
                              sampler=train_sampler,
                              num_workers=0)

    dev_out = preprocess.get_out(processor, './data/tianchi/dev.txt', args, label2id, 'dev')
    dev_features, dev_callback_info = dev_out
    dev_dataset = dataset.MLDataset(dev_features)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.eval_batch_size,
                            num_workers=0)

    trainer = Trainer(args, train_loader, dev_loader, dev_loader)
    # 训练和验证
    trainer.train()

    # 测试
    logger.info('========进行测试========')
    checkpoint_path = './checkpoints/best.pt'
    total_loss, test_outputs, test_targets = trainer.test(checkpoint_path)
    accuracy, micro_f1, macro_f1 = trainer.get_metrics(test_outputs, test_targets)
    logger.info(
        "【test】 loss：{:.6f} accuracy：{:.4f} micro_f1：{:.4f} macro_f1：{:.4f}".format(total_loss, accuracy, micro_f1,
                                                                                    macro_f1))
    report = trainer.get_classification_report(test_outputs, test_targets, labels)
    logger.info(report)

    # 预测
    trainer = Trainer(args, None, None, None)
    checkpoint_path = './checkpoints/best.pt'
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    predict_txt()
