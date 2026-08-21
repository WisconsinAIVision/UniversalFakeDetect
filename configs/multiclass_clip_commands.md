# Multiclass CLIP Commands

Set these paths for your machine:

```bash
DATA_ROOT=/path/to/dataset/root
TRAIN_CSV=/path/to/train.csv
VAL_CSV=/path/to/val.csv
CLASSES=real_good,real_bad,fake_good,fake_bad,manipulated_bad
```

## Frozen Backbone

```bash
python train.py --name=food_clip_vit_b32_frozen --data_mode=csv_multiclass --data_root=$DATA_ROOT --train_csv=$TRAIN_CSV --val_csv=$VAL_CSV --class_names=$CLASSES --arch=CLIP:ViT-B/32 --fix_backbone
python train.py --name=food_clip_vit_b16_frozen --data_mode=csv_multiclass --data_root=$DATA_ROOT --train_csv=$TRAIN_CSV --val_csv=$VAL_CSV --class_names=$CLASSES --arch=CLIP:ViT-B/16 --fix_backbone
python train.py --name=food_clip_vit_l14_frozen --data_mode=csv_multiclass --data_root=$DATA_ROOT --train_csv=$TRAIN_CSV --val_csv=$VAL_CSV --class_names=$CLASSES --arch=CLIP:ViT-L/14 --fix_backbone
```

## Full Fine-Tuning

```bash
python train.py --name=food_clip_vit_b32_finetune --data_mode=csv_multiclass --data_root=$DATA_ROOT --train_csv=$TRAIN_CSV --val_csv=$VAL_CSV --class_names=$CLASSES --arch=CLIP:ViT-B/32 --lr=1e-4 --backbone_lr=1e-6
python train.py --name=food_clip_vit_b16_finetune --data_mode=csv_multiclass --data_root=$DATA_ROOT --train_csv=$TRAIN_CSV --val_csv=$VAL_CSV --class_names=$CLASSES --arch=CLIP:ViT-B/16 --lr=1e-4 --backbone_lr=1e-6
python train.py --name=food_clip_vit_l14_finetune --data_mode=csv_multiclass --data_root=$DATA_ROOT --train_csv=$TRAIN_CSV --val_csv=$VAL_CSV --class_names=$CLASSES --arch=CLIP:ViT-L/14 --lr=1e-4 --backbone_lr=1e-6
```

Validation metrics are appended to:

```text
checkpoints/<experiment_name>/val_metrics.jsonl
```
