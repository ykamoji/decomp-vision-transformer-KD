# Attribution distillation learning for image classification 

### Requirements
```angular2html
torch==2.2.0
transformers==4.38.2
evaluate==0.4.1
```

### Fine-Tuning commands
```bash 
python3 fineTuning.py --model google/vit-base-patch16-224-in21k --train_batch_size 32 --val_batch_size 32 --epochs 1 --dataset cifar10 --dataset_dir ${data_path} 
```

### Distillation commands

-  Learn pretrained distilled models
```bash 
python3 distillation.py --model google/vit-base-patch16-224-in21k --student_model facebook/deit-base-distilled-patch16-224 --train_batch_size 32 --val_batch_size 32 --epochs 1 --distillation_token True --distillation_type soft --dataset cifar10 --dataset_dir ${data_path}
```

-  Learn pretrained non-distilled models
```bash 
python3 distillation.py --model google/vit-base-patch16-224-in21k --student_model facebook/deit-base-patch16-224 --train_batch_size 32 --val_batch_size 32 --epochs 1 --distillation_token False --distillation_type hard --dataset cifar10 --dataset_dir ${data_path} 
```