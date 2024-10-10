import evaluate
import torch
import json
import yaml
from datasets import load_dataset
from utils.argUtils import CustomObject, get_yaml_loader
from transformers import ViTImageProcessor, DeiTImageProcessor
from PIL import Image


with open('config.yaml', 'r') as file:
    config = yaml.load(file, get_yaml_loader())

Args = json.loads(json.dumps(config), object_hook=lambda d: CustomObject(**d))


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }


def collate_imageNet_fn(batch):
    return {
        'inputPath': [x['inputPath'] for x in batch],
        'labels': torch.tensor([x['label'] for x in batch])
    }


def collate_ImageNet_fine_tuning_fn(batch):

    collated_inputs = collate_imageNet_fn(batch)

    return processInputs(collated_inputs, Args.FineTuning.Model)


def processInputs(inputs, Model):

    if 'deit' in Model.Name:
        feature_extractor = DeiTImageProcessor.from_pretrained(Model.Name, cache_dir=Model.CachePath)
    else:
        feature_extractor = ViTImageProcessor.from_pretrained(Model.Name, cache_dir=Model.CachePath)

    images = [Image.open(Args.Common.DataSet.Path + '/' + path) for path in inputs['inputPath']]
    batches = [img.convert("RGB") if img.mode != 'RGB' else img for img in images]
    image_inputs = feature_extractor(batches, return_tensors='pt')
    return {
        'pixel_values': image_inputs['pixel_values'],
        'labels': inputs['labels']
    }


def build_metrics(metric_args):
    metrics_to_evaluate = metric_args.Name.split(',')
    for m in metrics_to_evaluate:
        _ = evaluate.load('custom_metrics/' + m, cache_dir=metric_args.CachePath, trust_remote_code=True)

    # accuracy = evaluate.load("accuracy", cache_dir='metrics/', trust_remote_code=True)

    metric = evaluate.combine(['custom_metrics/' + m for m in metrics_to_evaluate])

    def compute_metrics(p):
        return metric.compute(
            predictions=p.predictions,
            references=p.label_ids,
            labels=list(range(p.predictions.shape[1])),
        )

    return compute_metrics


def build_dataset(is_train, Args, show_details=True):
    DataSet = Args.Common.DataSet
    if Args.FineTuning.Action:
        Model = Args.FineTuning.Model
    elif Args.Distillation.Action:
        Model = Args.Distillation.Model
    else:
        Model = Args.Visualization.Model

    if 'deit' in Model.Name:
        feature_extractor = DeiTImageProcessor.from_pretrained(Model.Name, cache_dir=Model.CachePath)
    else:
        feature_extractor = ViTImageProcessor.from_pretrained(Model.Name, cache_dir=Model.CachePath)

    label_key = DataSet.Label

    if DataSet.Name == 'imageNet':
        def preprocess(batchImage):
            batches = [img.convert("RGB") if img.mode != 'RGB' else img for img in batchImage['image']]
            inputs = feature_extractor(batches, return_tensors='pt')
            inputs['label'] = batchImage[label_key]
            return inputs

    else:
        def preprocess(batchImage):
            inputs = feature_extractor(batchImage['img'], return_tensors='pt')
            inputs['label'] = batchImage[label_key]
            return inputs

    prepared_train = None
    if is_train:

        if DataSet.Name == 'imageNet':
            dataset_train = load_dataset('csv', split=f"train[:{DataSet.Train}]", verification_mode='no_checks',
                                         data_files={"train":DataSet.Path + "/metadata_train.csv"})
            prepared_train = dataset_train
        else:
            dataset_train = load_dataset(DataSet.Name, split=f"train[:{DataSet.Train}]", verification_mode='no_checks',
                                         cache_dir=DataSet.Path + "/train")
            prepared_train = dataset_train.with_transform(preprocess)

        num_training_labels = len(set(dataset_train[label_key]))

        if show_details:
            print(f"\nTraining info:{dataset_train}")
            print(f"\tNumber of labels = {num_training_labels}, {dataset_train.features[label_key]}")

    if DataSet.Name == 'imageNet':
        dataset_test = load_dataset('csv', split=f"validation[:{DataSet.Test}]",
                                    data_files={"validation":DataSet.Path + "/metadata_valid.csv"})

        prepared_test = dataset_test
    else:
        dataset_test = load_dataset(DataSet.Name, split=f"test[:{DataSet.Test}]", verification_mode='no_checks',
                                    cache_dir=DataSet.Path + "/test")

        prepared_test = dataset_test.with_transform(preprocess)

    num_validation_labels = len(set(prepared_test[label_key]))
    if show_details:
        print(f"\nTesting info:{dataset_test}")
        print(f"\tNumber of labels = {num_validation_labels}, {dataset_test.features[label_key]}")

    if is_train:
        return num_training_labels, prepared_train, prepared_test
    else:
        return num_validation_labels, None, prepared_test
