# Attribution distillation learning for image classification 

### Requirements
```angular2html
torch==2.2.0
transformers==4.38.2
evaluate==0.4.1
```

### Fine-Tuning commands
```bash 
python3 fineTuning.py --model google/vit-base-patch16-224-in21k --train_batch_size 32 --val_batch_size 32 --epochs 1 --lr 5e-05 --weight_decay 0.0 --dataset cifar10 --dataset_dir ${data_path} 
```

### Distillation commands

-  Learn pretrained distilled models
```bash 
python3 distillation.py --student_model facebook/deit-base-distilled-patch16-224 --batch_size 64 --epochs 10 --lr 0.0001 --weight_decay 0.001
--distillation_token True --distillation_type soft --dataset cifar10 --dataset_dir ${data_path}
```

-  Learn pretrained non-distilled models
```bash 
python3 distillation.py --student_model facebook/deit-base-patch16-224 --batch_size 64 --epochs 10 --lr 0.0001 --weight_decay 0.001
--distillation_token False --distillation_type hard --dataset cifar10 --dataset_dir ${data_path} --train 10% --test 10% 
```