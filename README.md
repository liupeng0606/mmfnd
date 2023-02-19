

## Dataset

Download the datasets

The Datasets of Fakeddit: [https://github.com/entitize/Fakeddit](https://github.com/entitize/Fakeddit)

The Datasets of Weibo: [https://github.com/yaqingwang/EANN-KDD18](https://github.com/yaqingwang/EANN-KDD18)

### (2) Extract images features

Generate the features for images

```shell script
python features_gen.py --dataset_dir ./data/weibo/ --image_dir ${weibo_image_dir} --feature_dir ./data/weibo/
python features_gen.py --dataset_dir ./data/fakeddit/ --image_dir ${fakeddit_image_dir} --feature_dir ./data/fakeddit/
```

## Train model

### (1) train model

```shell script
python train.py --task {task_name} --batch_sz 32 --gradient_accumulation_steps 24 --max_epochs 20 --bert_model {bert-base-uncased}
```




