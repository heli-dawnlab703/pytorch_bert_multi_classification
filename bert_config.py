import argparse


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):
        # args for path
        parser.add_argument('--output_dir', default='./checkpoints/',
                            help='the output dir for model checkpoints')
        parser.add_argument('--bert_label_hidden',type=int, default = 1024, help = ' bert rank model label hidden')

        parser.add_argument('--margin_dis',type=float, default = 10, help = 'margin loss para')

        #   ./model_hub/bert-base-chinese/     ./model_hub/roberta-wwm-ext-large/     ./model_hub/medbert-base-wwm-chinese/
        parser.add_argument('--bert_dir', default='./model_hub/bert-base-chinese/',
                            help='bert dir for uer')
        parser.add_argument('--data_dir', default='./data/final_data/',
                            help='data dir for uer')
        parser.add_argument('--log_dir', default='./logs/',
                            help='log dir for uer')

        # other args
        parser.add_argument('--num_tags', default=6, type=int,
                            help='number of tags')
        parser.add_argument('--seed', type=int, default=123, help='random seed')

        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')

        parser.add_argument('--max_seq_len', default=128, type=int)

        parser.add_argument('--eval_batch_size', default=16, type=int)

        parser.add_argument('--swa_start', default=3, type=int,
                            help='the epoch when swa start')

        parser.add_argument('--model_name', default="classification", type=str,
                            help='model name [rank,classification]')

        # train args
        parser.add_argument('--train_epochs', default=5, type=int,
                            help='Max training epoch')

        parser.add_argument('--dropout_prob', default=0.1, type=float,
                            help='drop out probability')

        # 2e-5
        parser.add_argument('--lr', default=3e-5, type=float,
                            help='learning rate for the bert module')
        # 2e-3
        parser.add_argument('--other_lr', default=3e-4, type=float,
                            help='learning rate for the module except bert')
        # 0.5
        parser.add_argument('--max_grad_norm', default=1, type=float,
                            help='max grad clip')

        parser.add_argument('--warmup_proportion', default=0.1, type=float)

        parser.add_argument('--weight_decay', default=0.01, type=float)

        parser.add_argument('--adam_epsilon', default=1e-8, type=float)

        parser.add_argument('--train_batch_size', default=16, type=int)

        parser.add_argument('--eval_model', default=True, action='store_true',
                            help='whether to eval model after training')

        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()
