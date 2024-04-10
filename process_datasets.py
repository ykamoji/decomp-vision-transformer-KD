from datasets import load_dataset
import evaluate
from transformers import ViTImageProcessor, DeiTImageProcessor
import torch
import numpy as np


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }


def build_metrics(metric_args):
    metrics_to_evaluate = metric_args.Name.split(',')
    for m in metrics_to_evaluate:
        _ = evaluate.load(m, cache_dir=metric_args.CachePath, trust_remote_code=True)

    # accuracy = evaluate.load("accuracy", cache_dir='metrics/', trust_remote_code=True)

    metric = evaluate.combine(metrics_to_evaluate)

    def compute_metrics(p):
        return metric.compute(
            predictions=np.argmax(p.predictions, axis=1),
            references=p.label_ids
        )

    return compute_metrics


def build_dataset(is_train, Args, show_details=True):
    DataSet = Args.Common.DataSet
    step = Args.FineTuning if Args.FineTuning.Action else Args.Distillation

    if 'deit' in step.Model.Name:
        feature_extractor = DeiTImageProcessor.from_pretrained(step.Model.Name, cache_dir=step.Model.CachePath)
    else:
        feature_extractor = ViTImageProcessor.from_pretrained(step.Model.Name, cache_dir=step.Model.CachePath)
    label_key = DataSet.Label

    def preprocess(batchImage):
        inputs = feature_extractor(batchImage['img'], return_tensors='pt')
        inputs['label'] = batchImage[label_key]
        return inputs

    prepared_train = None
    num_labels = 0
    if is_train:
        dataset_train = load_dataset(DataSet.Name, split=f"train[:{DataSet.Train}]", verification_mode='no_checks',
                                     cache_dir=DataSet.Path + "/train")

        num_labels = len(set(dataset_train[label_key]))

        if show_details:
            print(f"\nTraining info:{dataset_train}")
            print(f"\nNumber of labels = {num_labels}, {dataset_train.features[label_key]}")

        prepared_train = dataset_train.with_transform(preprocess)

    dataset_test = load_dataset(DataSet.Name, split=f"test[:{DataSet.Test}]", verification_mode='no_checks',
                                cache_dir=DataSet.Path + "/test")

    prepared_test = dataset_test.with_transform(preprocess)

    if show_details:
        print(f"\nTesting info:{dataset_test}")

    if is_train:
        return num_labels, prepared_train, prepared_test
    else:
        return prepared_test
