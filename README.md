# Attribution distillation learning for image classification 

### Requirements
```angular2html
torch==2.2.0
torchmetrics==1.3.1
torchvision==0.17.0
transformers==4.38.2
evaluate==0.4.1
numpy==1.24.4
re==2.2.1
tensorboard==2.16.2
pyyaml==6.0
json==2.0.9
seaborn==0.12.2
matplotlib==3.7.1
pandas==1.5.3
PIL==10.0.1
tqdm==4.65.0
p-tqdm==1.2
scipy==1.11.4
networkx==3.1
```

To start the fine-tuning and/or distillation actions, run the below command

<hr>

```bash 
python3 main.py 
```

<hr>

### Dataset configuration
- Update the Common.DataSet.Path manually or add environment variable ${DATASET_PATH}
- For cifar10, Common.DataSet.Label : label
- For cifar100, Common.DataSet.Label : fine_label  or coarse_label

<hr>

### Fine-Tuning configuration
-  FineTuning.Action: True

<hr>

### Distillation configuration
-  Distillation.Action: True
- To use models with distillation token, set UseDistTokens: True
- To use KL loss, set DistillationType: soft otherwise hard

<hr>

### Attribution visualization configuration
- Visualization.Action: True

<hr style="height: 0.5px">

### Results structure

    .
    ├──...     
    ├── Results                
    │    ├── FineTuned                
    │    │   ├── <Model Name>                      
    │    │   │    ├── <DataSet>                   
    │    │   │    │    ├── run_{index}                  
    │    │   │    │    │    ├── logs
    │    │   │    │    │    ├── training
    │    │   │    │    │    ├── tuned-model
    │    ├── Distilled                
    │    │    ├── <Model Name>                      
    │    │    │    ├── <DataSet>                   
    │    │    │    │    ├── run_{index}                  
    │    │    │    │    │    ├── logs
    │    │    │    │    │    ├── training
    │    │    │    │    │    ├── distilled-model


