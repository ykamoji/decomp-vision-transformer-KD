import scipy
import os
import csv
import time
import yaml
import json
import math
import threading
import random
from tqdm import tqdm
from utils.argUtils import CustomObject, get_yaml_loader
from typing import Optional, Any

BATCH_SIZE = 1000


def scrub_data(dataSetPath):

    imagePaths = []
    train_count, valid_count = 0, 0
    print("Collecting class names...")
    for root, dirs, files in os.walk(f"{dataSetPath}"):
        # imagePaths.extend([folder.split('.tar')[0] for folder in files if '.tar' in folder])

        if 'train' in root:
            imgPath = ["/".join([root.split('/')[-2], root.split('/')[-1]]) + '/' +
                       file for file in files if '.JPEG' in file]

            train_count += len(imgPath)
        else:
            imgPath = [root.split('/')[-1] + '/' + file for file in files if '.JPEG' in file]
            valid_count += len(imgPath)

        imagePaths.extend(imgPath)
    imagePaths = list(set(imagePaths))
    print(f"Crawled data:\n\tTraining: {train_count}\n\tValidation: {valid_count}")
    print("Creating metadata...")

    return imagePaths


def create_mappings(dataSetPath):

    with open(f"{dataSetPath}/label2id.json", 'r') as f:
        label2id = json.load(f)
    mat = scipy.io.loadmat(f"{dataSetPath}/meta.mat")
    id2label = {}
    validation_truth_map = {}
    for item in mat['synsets']:
        id2label[item[0][1][0]] = item[0][2][0]
        validation_truth_map[item[0][0][0][0]] = item[0][2][0]
    with open(f"{dataSetPath}/ILSVRC2012_validation_ground_truth.txt", 'r') as f:
        data = f.readlines()
    ground_truth_map = {}
    for idx, gt in enumerate(data):
        ground_truth_map[idx + 1] = gt.removesuffix('\n')

    return id2label, label2id, validation_truth_map, ground_truth_map


def get_label_frequency(val, data_to_write, split):

    labels = [data[1] for data in data_to_write if data[0].startswith(split)]
    unique_labels = list(set(labels))
    label_frequency = {key: math.ceil(labels.count(key) * val) for key in tqdm(unique_labels)}
    return label_frequency


def process_metadata(dataSetPath):

    imagePaths = scrub_data(dataSetPath)

    id2label, label2id, validation_truth_map, ground_truth_map = create_mappings(dataSetPath)

    class MetadataCreate(threading.Thread):

        def __init__(self, tid, begin, end) -> None:
            super().__init__()
            self.tid = tid
            self.begin = begin
            self.end = end
            self.local_mapping = []

        def run(self) -> None:
            # print(f"[START] {self.tid}")
            try:
                imageBatch = imagePaths[self.begin: self.end]
                for imgFile in imageBatch:
                    if 'train/' in imgFile:
                        label_key = id2label[imgFile.split('/')[-1].split('_')[0]]
                    else:
                        label_key = validation_truth_map[
                            int(ground_truth_map[int(imgFile.split('_')[-1].split('.')[0])])]

                    label = label2id[label_key]
                    self.local_mapping.append([imgFile, label])

            except Exception as e:
                print(e)

        def join(self, timeout: Optional[float] = ...) -> list[Any]:
            # print(f"[END] {self.tid}")
            return self.local_mapping

    batches = math.ceil(len(imagePaths) / BATCH_SIZE)
    print(f"Total batches = {batches}")
    MetadataCreateThreads = []
    start = time.time()
    for i in range(batches):
        create_thread = MetadataCreate(i + 1, i * BATCH_SIZE, (i + 1) * BATCH_SIZE)
        create_thread.start()
        MetadataCreateThreads.append(create_thread)
        # MetadataCreateThreads.append((i + 1, i * BATCH_SIZE, (i + 1) * BATCH_SIZE))
    data_to_write = []
    for create_thread in tqdm(MetadataCreateThreads):
        data_to_write.extend(create_thread.join())
    # print(MetadataCreateThreads)
    print(f"Time taken [Collection] = {((time.time() - start) / 60):.5f} seconds")
    return data_to_write


def write_metadata(dataSetPath, data_to_write):
    for dataset in ["", "_train", "_valid"]:
        with open(f"{dataSetPath}/metadata{dataset}.csv", 'a', newline='') as metadata:
            writer = csv.writer(metadata)
            writer.writerow(["inputPath", "label"])
    start = time.time()
    train_count, valid_count = 0, 0
    for data in tqdm(data_to_write):
        split = "train" if data[0].startswith("train") else "valid"
        with open(f"{dataSetPath}/metadata_{split}.csv", 'a', newline='') as metadata:
            writer = csv.writer(metadata)
            writer.writerow(data)

        if split == "train":
            train_count += 1
        else:
            valid_count += 1
    with open(f"{dataSetPath}/metadata.csv", 'a', newline='') as metadata:
        writer = csv.writer(metadata)
        writer.writerows(data_to_write)
    print(f"Time taken [Writing] = {((time.time() - start) / 60):.5f} seconds")
    # start = time.time()
    # progress = 0
    # total = len(imagePaths)
    # with open(f"{dataSet_path}/metadata.csv", 'w', newline='') as metadata:
    #     writer = csv.writer(metadata)
    #     writer.writerow(["file_name", "label"])
    #     for class_label in imagePaths:
    #         if 'train/' in class_label:
    #             label = id2label[class_label.split('/')[-1].split('_')[0]]
    #         else:
    #             label = validation_truth_map[int(ground_truth_map[int(class_label.split('_')[-1].split('.')[0])])]
    #
    #         writer.writerow([class_label, label2id[label]])
    #
    #         progress += 1
    #         if progress % 100 == 0:
    #             print(f"Completed {progress * 100 / total:.3f} %", end='\r', flush=True)
    #
    # print(f"Time taken [Writing] = {((time.time() - start) / 60):.5f} seconds")
    print("Metadata created !")
    print(f"Prepaid data:\n\tTraining: {train_count}\n\tValidation: {valid_count}")


def limit_per_class(Metadata, data_to_write):
    print(f"Processing label frequencies ...")
    start = time.time()
    train_label_frequency = get_label_frequency(Metadata.Value, data_to_write, "train")
    valid_label_frequency = get_label_frequency(Metadata.Value, data_to_write, "valid")
    label_frequency = {"train": train_label_frequency, "valid": valid_label_frequency}
    print(f"Time taken [Label frequency mapping] = {((time.time() - start) / 60):.5f} seconds")
    start = time.time()

    filtered_data_to_write = []
    for data in tqdm(data_to_write):
        split = "train" if data[0].startswith("train") else "valid"
        allow = False
        if label_frequency[split][data[1]] > 0:
            allow = True
            label_frequency[split][data[1]] -= 1

        if allow:
            filtered_data_to_write.append(data)
    print(f"Time taken [Filtering] = {((time.time() - start) / 60):.5f} seconds")
    return filtered_data_to_write


def limit_on_total(Metadata, data_to_write):
    filtered_data_to_write = []
    print(f"Processing total labels ...")
    start = time.time()
    labels = [data[1] for data in tqdm(data_to_write)]
    unique_labels = list(set(labels))
    print(f"Time taken [Labels limits] = {((time.time() - start) / 60):.5f} seconds")
    limit = Metadata.Value * len(unique_labels)
    start = time.time()
    for data in tqdm(data_to_write):
        if data[1] < limit:
            filtered_data_to_write.append(data)

    print(f"Time taken [Filtering] = {((time.time() - start) / 60):.5f} seconds")

    # filtered_labels = [data[1] for data in tqdm(filtered_data_to_write)]
    # label_frequency = {key: filtered_labels.count(key) for key in tqdm(unique_labels)}
    # print(label_frequency)

    return filtered_data_to_write


def create_metadata(dataSetPath, Metadata):
    if not os.path.exists(f"{dataSetPath}/metadata.csv") or \
            not os.path.exists(f"{dataSetPath}/metadata_train.csv") or \
            not os.path.exists(f"{dataSetPath}/metadata_valid.csv"):

        data_to_write = process_metadata(dataSetPath)

        ## Filtering when limit = True
        if Metadata.Limit:
            ## Sorting
            start = time.time()
            data_to_write = sorted(data_to_write, key=lambda data: data[0])
            print(f"Time taken [Sorting] = {((time.time() - start) / 60):.5f} seconds")

            # data_to_write = limit_per_class(Metadata, data_to_write)
            data_to_write = limit_on_total(Metadata, data_to_write)

        ## Randomizing
        start = time.time()
        random.seed(40)
        random.shuffle(data_to_write)
        print(f"Time taken [Randomizing] = {((time.time() - start) / 60):.5f} seconds")

        write_metadata(dataSetPath, data_to_write)



if __name__ == '__main__':

    with open('config.yaml', 'r') as file:
        config = yaml.load(file, get_yaml_loader())

    x = json.dumps(config)
    Args = json.loads(x, object_hook=lambda d: CustomObject(**d))

    create_metadata(Args.Common.DataSet.Path, Args.Metadata)
