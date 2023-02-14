
# Public source code of Multi-modal fake news detection via bridging the gap between modals



## Dataset


You need to download the original datasets from the following links to obtain image files.

Fakeddit: [https://github.com/entitize/Fakeddit](https://github.com/entitize/Fakeddit)

Weibo: [https://github.com/yaqingwang/EANN-KDD18](https://github.com/yaqingwang/EANN-KDD18)

### (2) Extract images features

```shell script
python extract_image_features.py --dataset_dir ./data/weibo/ --image_dir ${your_weibo_image_dir} --feature_dir ./data/weibo/
python extract_image_features.py --dataset_dir ./data/fakeddit/ --image_dir ${your_fakeddit_image_dir} --feature_dir ./data/fakeddit/
```

## Running

### (1) train model

```shell script
python train.py --task {task_name} --label_type 6_way_label --batch_sz 32 --gradient_accumulation_steps 20 --max_epochs 20 --name fakeddit_6_way --bert_model bert-base-uncased --global_image_embeds 5 --region_image_embeds 20 --num_image_embeds 25
```




