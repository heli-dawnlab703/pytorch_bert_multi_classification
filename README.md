# 启动笔记 -- > 2023-9-28
1. 可以运行run.sh
2. checkpoints 放置预训练好的模型  
3. 需要从huggingface中下载模型, 下载对应模型 中文的对应bert_base_chiese, 并放置model_hub目录中!
[link](https://huggingface.co/bert-base-chinese/tree/main)

# error
1. GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.git
[解决方法](https://blog.csdn.net/qq_39564555/article/details/100051051)

# 调优

## bert_base_chinese
--lr=3e-5  --model_name rank --other_lr=3e-4 --train_batch_size=64 --eval_batch_size=64 --train_epochs=11
--lr=3e-6 --model_name rank --other_lr=3e-4 --train_batch_size=32 --eval_batch_size=32 --train_epochs=11 --bert_dir ./model_hub/bert-base-chinese


## roberta-wwm-ext-large
--lr=3e-5  --model_name rank --other_lr=3e-4 --train_batch_size=64 --eval_batch_size=64 --train_epochs=11
--lr=3e-6 --model_name rank --other_lr=3e-4 --train_batch_size=16 --eval_batch_size=16 --train_epochs=11 --bert_dir ./model_hub/roberta-wwm-ext-large
分类: 不太行 阈值为0.5
--lr=3e-5  --model_name rank --other_lr=3e-4 --train_batch_size=64 --eval_batch_size=64 --train_epochs=11
【test】 loss：607.445513 accuracy：0.8530 micro_f1：0.8530 macro_f1：0.8180