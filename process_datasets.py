from datasets import load_dataset
import evaluate
from transformers import ViTImageProcessor, DeiTImageProcessor
import torch
import scipy
import json
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

        mat = scipy.io.loadmat(f"{DataSet.Path}/meta.mat")
        id2label = {}
        validation_truth_map = {}
        for item in mat['synsets']:
            id2label[item[0][1][0]] = item[0][2][0]
            validation_truth_map[item[0][0][0][0]] = item[0][2][0]

        with open(f"{DataSet.Path}/ILSVRC2012_validation_ground_truth.txt", 'r') as f:
            data = f.readlines()

        ground_truth_map = {}
        for idx, gt in enumerate(data):
            ground_truth_map[idx + 1] = gt.removesuffix('\n')

        with open(f"{DataSet.Path}/label2id.json", 'r') as f:
            label2id = json.load(f)

        def preprocess(batchImage):

            batches = [img.convert("RGB") if img.mode != 'RGB' else img for img in batchImage['image']]

            inputs = feature_extractor(batches, return_tensors='pt')

            labels_intermediate = []
            for image in batchImage['image']:
                filename = image.filename.split('/')[-1]
                if 'val' in filename:
                    labels_intermediate.append(
                        validation_truth_map[int(ground_truth_map[int(filename.split('_')[-1].split('.')[0])])])
                else:
                    labels_intermediate.append(id2label[filename.split('_')[0]])

            labels = [label2id[intermediate] for intermediate in labels_intermediate]

            inputs['label'] = labels
            return inputs
    else:
        def preprocess(batchImage):
            inputs = feature_extractor(batchImage['img'], return_tensors='pt')
            inputs['label'] = batchImage[label_key]
            return inputs

    prepared_train = None
    num_labels = 0
    if is_train:

        if DataSet.Name == 'imageNet':
            dataset_train = load_dataset('imagefolder', split=f"train[:{DataSet.Train}]", verification_mode='no_checks',
                                         data_dir=DataSet.Path)
            num_labels = 1000
        else:
            dataset_train = load_dataset(DataSet.Name, split=f"train[:{DataSet.Train}]", verification_mode='no_checks',
                                         cache_dir=DataSet.Path + "/train")

            num_labels = len(set(dataset_train[label_key]))

        if show_details:
            print(f"\nTraining info:{dataset_train}")
            print(f"\nNumber of labels = {num_labels}, {dataset_train.features[label_key]}")

        prepared_train = dataset_train.with_transform(preprocess)

    if DataSet.Name == 'imageNet':
        dataset_test = load_dataset('imagefolder', split=f"validation[:{DataSet.Test}]", verification_mode='no_checks',
                                    data_dir=DataSet.Path)
        num_labels = 1000
    else:
        dataset_test = load_dataset(DataSet.Name, split=f"test[:{DataSet.Test}]", verification_mode='no_checks',
                                    cache_dir=DataSet.Path + "/test")

    prepared_test = dataset_test.with_transform(preprocess)

    if show_details:
        print(f"\nTesting info:{dataset_test}")
        print(f"\nNumber of labels = {num_labels}, {dataset_test.features[label_key]}")

    if is_train:
        return num_labels, prepared_train, prepared_test
    else:
        return num_labels, None, prepared_test
