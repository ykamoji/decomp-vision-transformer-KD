from datasets import load_dataset
import evaluate
from transformers import ViTImageProcessor
import torch
import numpy as np



def collate_fn(batch):

    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }


def build_metrics(args):
    metrics_to_evaluate = args.metrics.split(',')
    for m in metrics_to_evaluate:
        _ = evaluate.load(m, cache_dir=args.metrics_dir, trust_remote_code=True)

    # accuracy = evaluate.load("accuracy", cache_dir='metrics/', trust_remote_code=True)

    metric = evaluate.combine(metrics_to_evaluate)

    def compute_metrics(p):
        return metric.compute(
            predictions=np.argmax(p.predictions, axis=1),
            references=p.label_ids
        )

    return compute_metrics


def build_dataset(is_train, args, show_details=True):
    feature_extractor = ViTImageProcessor.from_pretrained(args.model, cache_dir=args.model_dir)

    key = args.dataset_labels

    def preprocess(batchImage):
        inputs = feature_extractor(batchImage['img'], return_tensors='pt')
        inputs['label'] = batchImage[key]
        return inputs

    prepared_train = None
    num_labels = 0
    if is_train:
        dataset_train = load_dataset(args.dataset, split=f"train[:{args.train}]", verification_mode='no_checks',
                                     cache_dir=args.dataset_dir + "/train")


        num_labels = len(set(dataset_train[key]))

        if show_details:
            print(f"\nTraining info:{dataset_train}")
            print(f"\nNumber of labels = {num_labels}, {dataset_train.features[key]}")

        prepared_train = dataset_train.with_transform(preprocess)

    dataset_test = load_dataset(args.dataset, split=f"test[:{args.test}]", verification_mode='no_checks',
                                cache_dir=args.dataset_dir + "/test")

    prepared_test = dataset_test.with_transform(preprocess)

    if show_details:
        print(f"\nTesting info:{dataset_test}")

    if is_train:
        return num_labels, prepared_train, prepared_test
    else:
        return prepared_test
