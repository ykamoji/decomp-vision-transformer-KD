import torch
import pandas as pd
import seaborn as sns
import glob
import numpy as np
import warnings
import matplotlib.patches as pat
import matplotlib.pyplot as plt
from models_utils import ViTForImageClassification, DeiTForImageClassificationWithTeacher
from transformers import ViTImageProcessor, DeiTImageProcessor
from transformers.image_transforms import resize
from transformers.image_utils import PILImageResampling
from utils.featureUtils import process_features, feature_score ,mask_image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image


warnings.filterwarnings('ignore')


def visualize(Args):
    outputPath = Args.Visualization.Output
    showImages = Args.Visualization.Show
    saveImages = Args.Visualization.Save

    images = glob.glob(f"{Args.Visualization.Input}/*.JPEG")
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

    model.eval()
    model.to(device)
    factor = 14

    for im in images:

        label = im.split('_')[1].split('.')[0]
        image = Image.open(im)

        inputs = processor(images=image, return_tensors="pt")
        inputs.to(device)
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True, output_norms=True,
                        output_globenc=True)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        print(f"Actual: {label.ljust(10, ' ')} Predicted: {model.config.id2label[predicted_class_idx]}")

        trans_features = []
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,7))
        for features, feature_type, ax in zip([outputs.attributions, outputs.attentions],["Attribution", "Attention"],
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

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,7))

        img_resized = resize(np.array(image), size=(224, 224), resample=PILImageResampling.BILINEAR)
        grid_size = 224 // factor

        grid_color = [0, 0, 0]
        img_resized_grid = img_resized.copy()
        img_resized_grid[:, ::grid_size, :] = grid_color
        img_resized_grid[::grid_size, :, :] = grid_color

        ax1.imshow(img_resized_grid)
        ax1.axis('off')

        img_resized_final = img_resized.copy()
        for patches, feature_type, ax in zip([trans_features[0], trans_features[1]],["Attribution", "Attention"], [ax2, ax3]):

            attribute_score_per_patch = feature_score(patches)

            threshold_score = 3
            ax1.imshow(img_resized_final)
            if factor > 14:
                factor = 14
            x, y = np.meshgrid(np.arange(0, factor), np.arange(0, factor), indexing='ij')

            for i in range(factor):
                for j in range(factor):
                    index = x[i, j] * factor + y[i, j]
                    score = attribute_score_per_patch[index]
                    if score >= threshold_score:
                        if score == 1:
                            edgecolor = 'blue'
                            alpha = 0.2
                        elif score == 2:
                            edgecolor = 'yellow'
                            alpha = 0.4
                        elif score == 3:
                            edgecolor = 'orange'
                            alpha = 0.6
                        elif score == 4:
                            edgecolor = 'orangered'
                            alpha = 0.7
                        else:
                            edgecolor = 'red'
                            alpha = 1

                        rect = pat.Rectangle((y[i, j] * grid_size, x[i, j] * grid_size), grid_size - 1, grid_size - 1,
                                             linewidth=1, edgecolor=edgecolor, facecolor='none', alpha=alpha)

                        ax.add_patch(rect)
                        ax.imshow(img_resized_final)
                        ax.set_title(f"{feature_type}")
                        ax.axis('off')

        if saveImages:
            title = f"Actual: {label}, Predicted: {model.config.id2label[predicted_class_idx]}"
            fig.suptitle(title, y=0.9, size=15)
            plt.savefig(outputPath + f"{label}_features")
        if showImages:
            plt.show()

    if Args.Visualization.Plot.PlotMaskedCurves:
        plotMaskedCurves(model, processor, images, Args)


def eval_model(dataset, model, strategy, Args):
    dataloader = DataLoader(dataset, batch_size=Args.Visualization.Plot.BatchSize)
    pbar = tqdm(iter(dataloader))
    progress, masking_accuracy = 0, 0
    for batch in pbar:
        inputs = {'pixel_values':batch['pixel_values'].to(Args.Visualization.Model.Device)}
        outputs = model(**inputs)
        logits = outputs.logits
        labels = batch['label']
        preds = logits.argmax(-1)
        for i in range(len(labels)):
            progress += 1
            # print(f"Actual: {labels[i].ljust(10, ' ')} Predicted: {model.config.id2label[preds[i].item()]}")
            masking_accuracy += labels[i] in model.config.id2label[preds[i].item()]
            pbar.set_postfix({"Accuracy": f"{masking_accuracy/progress:.3f}"})

    masking_accuracy /= len(dataset)
    print(f"{strategy} Masking Accuracy: {masking_accuracy*100:.3f} %")

def plotMaskedCurves(model, processor, images, Args):
    dataset = []
    pbar = tqdm(images[:16])
    pbar.set_description("Image Masking")
    images_downstream = []
    original_images = []
    for im in pbar:
        label = im.split('_')[1].split('.')[0]
        image = Image.open(im)
        image_nd = np.array(image)
        if image_nd.ndim < 3:
            continue
        processed_image = processor(images=image, return_tensors="pt")
        processed_image = processed_image.data['pixel_values'][0]
        images_downstream.append({'pixel_values':processed_image, 'label':label})
        original_images.append(image_nd)
        masked_image = mask_image(processed_image, type='random', mask_perc=Args.Visualization.Plot.MaskedPerc)
        input = {'pixel_values': masked_image, 'label':label}
        dataset.append(input)

    eval_model(dataset, model, "random", Args)

    mask_attention_plot(model, images_downstream, original_images, Args)

    ##TODO :: masking based on attribution

    ##TODO :: masking based on ATS score


def mask_attention_plot(model, dataset, original_images, Args):

    dataloader = DataLoader(dataset, batch_size=Args.Visualization.Plot.BatchSize)
    pbar = tqdm(iter(dataloader))
    pbar.set_description("Attention Scores")
    attention_scores = []
    for batch in pbar:
        inputs = {'pixel_values':batch['pixel_values'].to(Args.Visualization.Model.Device)}
        outputs = model(**inputs, output_attentions=True)
        attention_scores += process_attention_batch(outputs.attentions)

    masked_attn_dataset = []
    pbar = tqdm(range(len(dataset)))
    pbar.set_description("Image Masking (Attn)")
    for index in pbar:
        image = dataset[index]['pixel_values']
        scores = attention_scores[index]
        masked_attn_image = mask_image(image, type='Attention', mask_perc=Args.Visualization.Plot.MaskedPerc,
                                       scores=scores)

        masked_attn_dataset.append(masked_attn_image)

    eval_model(masked_attn_dataset, model, "Attention", Args)

def process_attention_batch(attentions):

    num_layers = len(attentions)

    batch_tensor = torch.stack([attentions[i] for i in range(num_layers)])
    batch_tensor = batch_tensor.permute((1, 0, 2, 3, 4))
    single_image_features = (batch_tensor[i] for i in range(batch_tensor.shape[0]))
    scores = []
    for image_attn in single_image_features:
        patches = process_features(image_attn, 14, featureType='Attention')
        attn_scores = feature_score(patches)
        scores.append(attn_scores)

    return scores