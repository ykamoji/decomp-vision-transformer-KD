import scipy
import os
import csv
import time
import yaml
import json
import math
import threading
from tqdm import tqdm
from utils.argUtils import CustomObject, get_yaml_loader
from typing import Optional, Any

BATCH_SIZE = 1000


def create_metadata(dataSet_path):

    if not os.path.exists(f"{dataSet_path}/metadata.csv") or \
            not os.path.exists(f"{dataSet_path}/metadata_train.csv") or \
            not os.path.exists(f"{dataSet_path}/metadata_valid.csv"):

        mat = scipy.io.loadmat(f"{dataSet_path}/meta.mat")
        id2label = {}
        validation_truth_map = {}
        for item in mat['synsets']:
            id2label[item[0][1][0]] = item[0][2][0]
            validation_truth_map[item[0][0][0][0]] = item[0][2][0]

        with open(f"{dataSet_path}/ILSVRC2012_validation_ground_truth.txt", 'r') as f:
            data = f.readlines()

        ground_truth_map = {}
        for idx, gt in enumerate(data):
            ground_truth_map[idx + 1] = gt.removesuffix('\n')

        with open(f"{dataSet_path}/label2id.json", 'r') as f:
            label2id = json.load(f)

        class_labels = []
        train_count, valid_count = 0, 0
        print("Collecting class names...")
        for root, dirs, files in os.walk(f"{dataSet_path}"):
            # class_labels.extend([folder.split('.tar')[0] for folder in files if '.tar' in folder])

            if 'train' in root:
                labels = ["/".join([root.split('/')[-2], root.split('/')[-1]]) + '/' +
                               file for file in files if '.JPEG' in file]

                train_count += len(labels)
            else:
                labels = [root.split('/')[-1] + '/' + file for file in files if '.JPEG' in file]
                valid_count += len(labels)

            class_labels.extend(labels)

        class_labels = list(set(class_labels))

        print(f"Crawled data:\n\tTraining: {train_count}\n\tValidation: {valid_count}")
        print("Creating metadata...")

        class MetadataCreate(threading.Thread):

            def __init__(self, tid, begin, end) -> None:
                super().__init__()
                self.tid = tid
                self.begin = begin
                self.end = end
                self.local_labels = []

            def run(self) -> None:
                # print(f"[START] {self.tid}")
                try:
                    labels = class_labels[self.begin: self.end]
                    for class_label in labels:
                        if 'train/' in class_label:
                            label = id2label[class_label.split('/')[-1].split('_')[0]]
                        else:
                            label = validation_truth_map[
                                int(ground_truth_map[int(class_label.split('_')[-1].split('.')[0])])]

                        self.local_labels.append([class_label, label2id[label]])

                except Exception as e:
                    print(e)

            def join(self, timeout: Optional[float] = ...) -> list[Any]:
                # print(f"[END] {self.tid}")
                return self.local_labels

        batches = math.ceil(len(class_labels) / BATCH_SIZE)
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

        print(f"Time taken [Collection] = {((time.time() - start) /60):.5f} seconds")

        ## sorting
        start = time.time()
        data_to_write = sorted(data_to_write, key=lambda data: data[0])
        print(f"Time taken [Sorting] = {((time.time() - start) / 60):.5f} seconds")

        for dataset in ["","_train","_valid"]:
            with open(f"{dataSet_path}/metadata{dataset}.csv", 'a', newline='') as metadata:
                writer = csv.writer(metadata)
                writer.writerow(["inputPath", "label"])

        start = time.time()
        for data in tqdm(data_to_write):
            split = "train" if data[0].startswith("train") else "valid"
            with open(f"{dataSet_path}/metadata_{split}.csv", 'a', newline='') as metadata:
                writer = csv.writer(metadata)
                writer.writerow(data)

        with open(f"{dataSet_path}/metadata.csv", 'a', newline='') as metadata:
            writer = csv.writer(metadata)
            writer.writerows(data_to_write)

        print(f"Time taken [Writing] = {((time.time() - start) / 60):.5f} seconds")

        # start = time.time()
        # progress = 0
        # total = len(class_labels)
        # with open(f"{dataSet_path}/metadata.csv", 'w', newline='') as metadata:
        #     writer = csv.writer(metadata)
        #     writer.writerow(["file_name", "label"])
        #     for class_label in class_labels:
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


if __name__ == '__main__':

    with open('config.yaml', 'r') as file:
        config = yaml.load(file, get_yaml_loader())

    x = json.dumps(config)
    Args = json.loads(x, object_hook=lambda d: CustomObject(**d))

    create_metadata(Args.Common.DataSet.Path)
