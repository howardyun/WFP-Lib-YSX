dataset=OW
defence_name=Org
defence_dataset=${dataset}_${defence_name}

PYTHONPATH=. /home/root/.conda/envs/WFP-ysx/bin/python -u exp/test.py \
  --dataset ${dataset} \
  --data_root ./defence_datesets \
  --data_dataset ${defence_dataset} \
  --result_root ./defence_datesets \
  --result_dataset ${defence_dataset} \
  --model AWF \
  --device cuda:0 \
  --feature DIR \
  --seq_len 3000 \
  --batch_size 256 \
  --eval_metrics Accuracy Precision Recall F1-score \
  --load_name max_f1
