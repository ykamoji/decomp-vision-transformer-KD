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
from utils.featureUtils import process_features, feature_score, mask_image, plot_feature_scores, \
    get_device, show_masked_images
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')


def visualize(Args):
    outputPath = Args.Visualization.Output

    images = glob.glob(f"{Args.Visualization.Input}/*/*.JPEG")

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
    label_map = {k: ", ".join(v) for k, v in data[0].items()}

    model.eval()
    model.to(get_device())
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
                # print(f"Skipping {im} due to incorrect image dimensions")
                continue
            try:
                inputs = processor(images=image, return_tensors="pt")
            except Exception as e:
                # print(f"Error: {e}\nSkipping {im}")
                continue

            inputs = inputs.to(get_device())

            outputs = model(**inputs, output_attentions=True, output_hidden_states=True, output_norms=False,
                              output_globenc=True, output_ats = 1 if Args.Visualization.UseOnlyCLSForATS else 2)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            print(f"Actual: {label.ljust(10, ' ')} Predicted: {model.config.id2label[predicted_class_idx]}", end=' ')

            if label in model.config.id2label[predicted_class_idx]:
                print(f" Correct!")
            else:
                print(f" Failed!")

            trans_features = []

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 7))
            for features, feature_type, ax in zip([outputs.attributions, outputs.attentions, outputs.ats_attentions],
                                                  ["Attribution", "Attention", "ATS"],
                                                  [ax1, ax2, ax3]):
                patches = process_features(features, factor, featureType=feature_type,
                                           strategies=Args.Visualization.Strategies)
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

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 8))

            img_resized = resize(np.array(image), size=(224, 224), resample=PILImageResampling.BILINEAR)
            grid_size = 224 // factor

            # grid_color = [0, 0, 0]
            # img_resized_grid = img_resized.copy()
            # img_resized_grid[:, ::grid_size, :] = grid_color
            # img_resized_grid[::grid_size, :, :] = grid_color

            ax1.imshow(img_resized)
            ax1.axis('off')

            img_resized_feature = img_resized.copy()
            for patches, feature_feature_type, ax in zip([trans_features[0], trans_features[1], trans_features[2]],
                                                 ["Attribution", "Attention", "ATS"],
                                                 [ax2, ax3, ax4]):

                attribute_score_per_patch = feature_score(patches)
                if factor > 14:
                    factor = 14

                plot_feature_scores(attribute_score_per_patch, ax, factor, feature_feature_type, grid_size, img_resized_feature,
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
            if K > 10: time.sleep(1)
            print(f"\n------ Evaluating for K={K}% -------\n")
            results[K] = plotMaskedCurves(model, processor, images, label_map, K, Args)

        accuracies = results[Args.Visualization.Plot.MaskedPerc[0]].keys()
        with open(f'{outputPath}/Accuracies.csv', 'w') as f:
            w = csv.DictWriter(f, accuracies)
            w.writeheader()
            for K in results.keys():
                w.writerow(results[K])

        markers = ['r', 'g', 'k', 'b']
        marker_idx = 0
        plt.style.use('seaborn-v0_8-darkgrid')
        for acc in accuracies:
            if acc == 'K':
                continue
            acc_list = [results[K][acc] for K in results.keys()]
            limit = Args.Visualization.Plot.MaskedPerc[-1] + 10
            plt.plot(np.arange(0, limit, 10), acc_list, markers[marker_idx], marker='', label=f"{acc}")
            marker_idx += 1
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Masking K %")
        # plt.title('Strategy: ' + ','.join(Args.Visualization.Strategies))
        plt.savefig(outputPath + f"masking_{','.join(Args.Visualization.Strategies)}_accuracies",
                    bbox_inches='tight', edgecolor='auto')
        plt.show()

    if Args.Visualization.Masking.Action:
        for index, im in enumerate(images):
            # label = im.split('_')[1].split('.')[0]
            label = label_map[im.split('/')[-2]]
            # label = label_map[im.split('/')[-1].split('_')[0]]
            image = Image.open(im)
            image_nd = np.array(image)
            for featureType in ["random", "Attention", "ATS", "Attribution"]:
                cache_name = f"feature_{featureType}_{','.join(Args.Visualization.Strategies)}_cache.npy"
                if os.path.exists(cache_name):
                    feature_scores_cache = np.load(cache_name)[index]
                else:
                    feature_scores_cache = None
                show_masked_images(image_nd, label, featureType=featureType, scores=feature_scores_cache,
                                   Args=Args, mask_percs=list(range(0, 60, 10)))

def plotMaskedCurves(model, processor, images, label_map, K, Args):
    dataset = []
    pbar = tqdm(images)
    pbar.set_description("Image Masking (Random)")
    images_downstream = []
    for im in pbar:
        # label = im.split('_')[1].split('.')[0]
        # label = label_map[im.split('/')[-2]]
        label = label_map[im.split('/')[-1].split('_')[0]]
        image = Image.open(im)
        image_nd = np.array(image)
        if image_nd.ndim < 3:
            # print(f"Skipping {im} due to incorrect image dimensions")
            continue
        try:
            processed_image = processor(images=image, return_tensors="pt")
        except Exception as e:
            # print(f"Error: {e}\nSkipping {im}")
            continue
        processed_image = processed_image.data['pixel_values'][0]
        images_downstream.append({'pixel_values': processed_image, 'label': label})
        masked_image = mask_image(processed_image, featureType='random', mask_perc=K)
        dataset.append({'pixel_values': masked_image, 'label': label})

    result = {"K": K}

    random_accuracy = eval_model(dataset, model, "Random", Args)

    result["random_accuracy"] = random_accuracy

    attention_accuracy = mask_feature_eval(images_downstream, model,
                                           'Attention',
                                           {"output_attentions": True},
                                           K,
                                           Args)

    result["attention_accuracy"] = attention_accuracy

    logic = 1 if Args.Visualization.UseOnlyCLSForATS else 2
    ats_accuracy = mask_feature_eval(images_downstream, model,
                                     'ATS',
                                     {"output_ats": logic},
                                     K,
                                     Args)
    result["ats_accuracy"] = ats_accuracy

    attribution_accuracy = mask_feature_eval(images_downstream, model,
                                             'Attribution',
                                             {"output_attentions": True, "output_hidden_states": True,
                                              "output_norms": False, "output_globenc": True},
                                             K,
                                             Args)

    result["attribution_accuracy"] = attribution_accuracy

    return result


def mask_feature_eval(dataset, model, feature_type, params, K, Args):
    cached = False
    cache_name = f"feature_{feature_type}_{','.join(Args.Visualization.Strategies)}_cache.npy"
    if os.path.exists(cache_name):
        feature_scores_cache = np.load(cache_name)
        cached = True

    if cached:
        feature_scores = feature_scores_cache.tolist()
    else:
        dataloader = DataLoader(dataset, batch_size=Args.Visualization.Plot.BatchSize)
        pbar = tqdm(iter(dataloader))
        pbar.set_description(f"Eval {feature_type} Scores")
        feature_scores = []
        for batch in pbar:
            inputs = {'pixel_values': batch['pixel_values'].to(get_device())}
            outputs = model(**inputs, **params)
            if feature_type == 'Attention':
                features = outputs.attentions
            elif feature_type == 'Attribution':
                features = outputs.attributions
            elif feature_type == 'ATS':
                features = outputs.ats_attentions
            feature_scores += process_feature_output_batch(features, feature_type, Args.Visualization.Strategies)

        feature_scores_nd = np.array(feature_scores)
        np.save(cache_name, feature_scores_nd)

    masked_attn_dataset = []
    pbar = tqdm(range(len(dataset)))
    pbar.set_description(f"Image Masking ({feature_type})")
    for index in pbar:
        image = dataset[index]['pixel_values']
        scores = feature_scores[index]
        masked_attn_image = mask_image(image, featureType=feature_type, mask_perc=K, scores=scores,
                                       threshold_score=Args.Visualization.Plot.ThresholdScore)
        masked_attn_image = {'pixel_values': masked_attn_image, 'label': dataset[index]['label']}
        masked_attn_dataset.append(masked_attn_image)

    return eval_model(masked_attn_dataset, model, feature_type, Args)


def process_feature_output_batch(features, feature_type, strategies):
    num_layers = len(features)

    if feature_type == "Attention" or feature_type == "ATS":
        batch_tensor = torch.stack([features[i] for i in range(num_layers)])
    elif feature_type == "Attribution":
        if type(features[0]) == tuple:
            batch_tensor = torch.stack([features[i][4] for i in range(num_layers)])
        else:
            batch_tensor = torch.stack([features[i] for i in range(num_layers)])

    shape = (1, 0,) + tuple(range(2, batch_tensor.ndim))
    batch_tensor = batch_tensor.permute(shape)
    single_image_features = (batch_tensor[i] for i in range(batch_tensor.shape[0]))
    scores = []
    for image_attn in single_image_features:
        patches = process_features(image_attn, 14, featureType=feature_type, strategies=strategies)
        attn_scores = feature_score(patches)
        scores.append(attn_scores)

    return scores


def eval_model(dataset, model, strategy, Args):
    dataloader = DataLoader(dataset, batch_size=Args.Visualization.Plot.BatchSize)
    pbar = tqdm(iter(dataloader))
    progress, masking_accuracy = 0, 0
    pbar.set_description(f"Eval {strategy}")
    for batch in pbar:
        inputs = {'pixel_values': batch['pixel_values'].to(get_device())}
        outputs = model(**inputs)
        logits = outputs.logits
        labels = batch['label']
        preds = logits.argmax(-1)
        for i in range(len(labels)):
            progress += 1
            # print(f"Actual: {labels[i].ljust(10, ' ')} Predicted: {model.config.id2label[preds[i].item()]}")
            masking_accuracy += labels[i] in model.config.id2label[preds[i].item()]
            pbar.set_postfix({"Accuracy": f"{masking_accuracy / progress:.7f}"})

    masking_accuracy /= len(dataset)
    # print(f"{strategy} Masking Accuracy: {masking_accuracy * 100:.7f} %")

    return round(masking_accuracy, 7)