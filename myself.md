# 高斯调参

```text
超参数是：nohup python -u main.py --lr 1e-5 
--model_name rank --other_lr=3e-4 
--train_batch_size=32 --eval_batch_size=32 
--train_epochs=20 --bert_label_hidden 768 --margin_dis 0.05 --max_seq_len 32 > rank.log  2>&1 &
```