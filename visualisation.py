import torch
import pandas as pd
import seaborn as sns
import glob
import numpy as np
import time
import csv
import warnings
from tqdm import tqdm
import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from models_utils import ViTForImageClassification, DeiTForImageClassificationWithTeacher
from transformers import ViTImageProcessor, DeiTImageProcessor
from transformers.image_transforms import resize
from transformers.image_utils import PILImageResampling
from utils.featureUtils import process_features, feature_score, mask_image, plot_feature_scores
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')


def visualize(Args):
    outputPath = Args.Visualization.Output

    images = glob.glob(f"{Args.Visualization.Input}/*/*.JPEG")
    device = Args.Visualization.Model.Device

    # model_path = get_model_path('FineTuned', Args)
    model_name = Args.Visualization.Model.Name
    if 'distilled' in model_name:
        classifier = DeiTForImageClassificationWithTeacher
        feature_extractor = DeiTImageProcessor
    else:
        classifier = ViTForImageClassification
        feature_extractor = ViTImageProcessor

    model = classifier.from_pretrained(model_name, cache_dir=Args.Visualization.Model.CachePath)

    processor = feature_extractor.from_pretrained(model_name, cache_dir=Args.Visualization.Model.CachePath)

    data = pickle.load(open(Args.Visualization.Input + f'/data.pkl', 'rb'), encoding='latin-1')

    label_map = {k: " ".join(v) for k, v in data[0].items()}

    model.eval()
    model.to(device)
    factor = 14

    if Args.Visualization.Features.CompareFeatures:

        showImages = Args.Visualization.Features.Show
        saveImages = Args.Visualization.Features.Save
        threshold_score = Args.Visualization.Features.ThresholdScore

        for im in images:

            # label = im.split('_')[1].split('.')[0]
            label = label_map[im.split('/')[-2]]

            image = Image.open(im)

            if np.array(image).ndim < 3:
                continue

            inputs = processor(images=image, return_tensors="pt")
            inputs.to(device)
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True, output_norms=True,
                            output_globenc=True)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            print(f"Actual: {label.ljust(10, ' ')} Predicted: {model.config.id2label[predicted_class_idx]}")

            trans_features = []

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))
            for features, feature_type, ax in zip([outputs.attributions, outputs.attentions],
                                                  ["Attribution", "Attention"],
                                                  [ax1, ax2]):
                patches = process_features(features, factor, featureType=feature_type)
                trans_features.append(patches)
                df = pd.DataFrame(patches, columns=np.arange(1, patches.shape[1] + 1), index=range(len(patches), 0, -1))
                sns.heatmap(df, cmap="Reds", square=False, ax=ax, cbar=False, xticklabels=False)
                bottom, top = ax.get_ylim()
                ax.set_ylim(bottom + 0.5, top - 0.5)
                ax.set_title(f"{feature_type}")

            if saveImages:
                plt.savefig(outputPath + f"{label}_heatmap")
            if showImages:
                plt.show()

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 7))

            img_resized = resize(np.array(image), size=(224, 224), resample=PILImageResampling.BILINEAR)
            grid_size = 224 // factor

            grid_color = [0, 0, 0]
            img_resized_grid = img_resized.copy()
            img_resized_grid[:, ::grid_size, :] = grid_color
            img_resized_grid[::grid_size, :, :] = grid_color

            ax1.imshow(img_resized_grid)
            ax1.axis('off')

            img_resized_feature = img_resized.copy()
            for patches, feature_type, ax in zip([trans_features[0], trans_features[1]], ["Attribution", "Attention"],
                                                 [ax2, ax3]):

                attribute_score_per_patch = feature_score(patches)

                ax1.imshow(img_resized_feature)
                if factor > 14:
                    factor = 14

                plot_feature_scores(attribute_score_per_patch, ax, factor, feature_type, grid_size, img_resized_feature,
                                    threshold_score)

            if saveImages:
                title = f"Actual: {label}, Predicted: {model.config.id2label[predicted_class_idx]}"
                fig.suptitle(title, y=0.9, size=15)
                plt.savefig(outputPath + f"{label}_features")
            if showImages:
                plt.show()

    if Args.Visualization.Plot.PlotMaskedCurves:

        results = {}

        for K in Args.Visualization.Plot.MaskedPerc:
            if K > 10: time.sleep(5)
            print(f"\n------ Evaluating for K={K}% -------\n")
            results[K] = plotMaskedCurves(model, processor, images, label_map, K, Args)

        accuracies = results[Args.Visualization.Plot.MaskedPerc[0]].keys()
        with open(f'{outputPath}/Accuracies.csv', 'w') as f:
            w = csv.DictWriter(f, accuracies)
            w.writeheader()
            for K in results.keys():
                w.writerow(results[K])


def plotMaskedCurves(model, processor, images, label_map, K, Args):
    dataset = []
    pbar = tqdm(images)
    pbar.set_description("Image Masking")
    images_downstream = []
    for im in pbar:
        # label = im.split('_')[1].split('.')[0]
        label = label_map[im.split('/')[-2]]
        image = Image.open(im)
        image_nd = np.array(image)
        if image_nd.ndim < 3:
            continue
        processed_image = processor(images=image, return_tensors="pt")
        processed_image = processed_image.data['pixel_values'][0]
        images_downstream.append({'pixel_values': processed_image, 'label': label})
        masked_image = mask_image(processed_image, type='random', mask_perc=K)
        dataset.append({'pixel_values': masked_image, 'label': label})

    random_accuracy = eval_model(dataset, model, "Random", Args)

    attention_accuracy = mask_feature_eval(images_downstream, model,
                                             'Attention',
                                             {"output_attentions": True},
                                             K,
                                             Args)

    attribution_accuracy = mask_feature_eval(images_downstream, model,
                                               'Attribution',
                                             {"output_attentions": True, "output_hidden_states": True,
                                                "output_norms": True, "output_globenc": True},
                                             K,
                                             Args)

    ##TODO :: masking based on ATS score

    return {"K": K, "random_accuracy": random_accuracy, "attention_accuracy": attention_accuracy,
             "attribution_accuracy":attribution_accuracy}



def mask_feature_eval(dataset, model, type, params, K, Args):

    cached = False
    if os.path.exists(f".feature_{type}_cache.npy"):
        feature_scores_cache = np.load(f".feature_{type}_cache.npy")
        cached = True

    if cached:
        feature_scores = feature_scores_cache.tolist()
    else:
        dataloader = DataLoader(dataset, batch_size=Args.Visualization.Plot.BatchSize)
        pbar = tqdm(iter(dataloader))
        pbar.set_description(f"Eval {type} Scores")
        feature_scores = []
        for batch in pbar:
            inputs = {'pixel_values': batch['pixel_values'].to(Args.Visualization.Model.Device)}
            outputs = model(**inputs, **params)
            if type == 'Attention':
                features = outputs.attentions
            elif type == 'Attribution':
                features = outputs.attributions
            feature_scores += process_feature_output_batch(features, type)

        feature_scores_nd = np.array(feature_scores)
        np.save(f".feature_{type}_cache", feature_scores_nd)

    masked_attn_dataset = []
    pbar = tqdm(range(len(dataset)))
    pbar.set_description(f"Image Masking ({type})")
    for index in pbar:
        image = dataset[index]['pixel_values']
        scores = feature_scores[index]
        masked_attn_image = mask_image(image, type=type, mask_perc=K, scores=scores,
                                       threshold_score=Args.Visualization.Plot.ThresholdScore)
        masked_attn_image = {'pixel_values': masked_attn_image, 'label': dataset[index]['label']}
        masked_attn_dataset.append(masked_attn_image)

    return eval_model(masked_attn_dataset, model, type, Args)


def process_feature_output_batch(features, type):
    num_layers = len(features)

    if type == "Attention":
        batch_tensor = torch.stack([features[i] for i in range(num_layers)])
        batch_tensor = batch_tensor.permute((1, 0, 2, 3, 4))
    elif type == "Attribution":
        batch_tensor = torch.stack([features[i][4] for i in range(num_layers)])
        batch_tensor = batch_tensor.permute((1, 0, 2, 3))

    single_image_features = (batch_tensor[i] for i in range(batch_tensor.shape[0]))
    scores = []
    for image_attn in single_image_features:
        patches = process_features(image_attn, 14, featureType=type)
        attn_scores = feature_score(patches)
        scores.append(attn_scores)

    return scores


def eval_model(dataset, model, strategy, Args):
    dataloader = DataLoader(dataset, batch_size=Args.Visualization.Plot.BatchSize)
    pbar = tqdm(iter(dataloader))
    progress, masking_accuracy = 0, 0
    pbar.set_description(f"Eval {strategy}")
    for batch in pbar:
        inputs = {'pixel_values': batch['pixel_values'].to(Args.Visualization.Model.Device)}
        outputs = model(**inputs)
        logits = outputs.logits
        labels = batch['label']
        preds = logits.argmax(-1)
        for i in range(len(labels)):
            progress += 1
            # print(f"Actual: {labels[i].ljust(10, ' ')} Predicted: {model.config.id2label[preds[i].item()]}")
            masking_accuracy += labels[i] in model.config.id2label[preds[i].item()]
            pbar.set_postfix({"Accuracy": f"{masking_accuracy / progress:.3f}"})

    masking_accuracy /= len(dataset)
    # print(f"{strategy} Masking Accuracy: {masking_accuracy * 100:.3f} %")

    return round(masking_accuracy, 3)