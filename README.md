# Attribution distillation learning for image classification 

### Requirements
```angular2html
torch==2.2.0
transformers==4.38.2
evaluate==0.4.1
pyyaml==6.0
json==2.0.9
```

To start the fine-tuning and/or distillation actions, run the below command

```bash 
python3 main.py 
```

### Dataset configuration changes
- Update the Common.DataSet.Path manually or add environment variable ${DATASET_PATH}
- For cifar10, Common.DataSet.Label : label
- For cifar100, Common.DataSet.Label : fine_label  or coarse_label

### Fine-Tuning configuation changes
-  FineTuning.Action: True


### Distillation configuration changes
-  Distillation.Action: True
- To use models with distillation token, set UseDistTokens: True
- To use KL loss, set DistillationType: soft otherwise hard


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


