from models_utils import ViTForImageClassification, DeiTForImageClassificationWithTeacher
from transformers import ViTImageProcessor, DeiTImageProcessor
from transformers.image_transforms import resize
from transformers.image_utils import PILImageResampling
from utils.featureUtils import process_features
import matplotlib.patches as pat
import matplotlib.pyplot as plt
import glob
import numpy as np
from PIL import Image
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


def visualize(Args):
    outputPath = Args.Visualization.Output
    showImages = Args.Visualization.Show
    saveImages = Args.Visualization.Save

    images = glob.glob(f"{Args.Visualization.Input}/*.JPEG")
    Device = Args.Visualization.Model.Device

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
    model.to(Device)

    for im in images:

        label = im.split('_')[1].split('.')[0]
        image = Image.open(im)
        factor = 14

        inputs = processor(images=image, return_tensors="pt")
        inputs.to(Device)
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

            mean = np.mean(patches[0])
            std = np.std(patches[0])

            attribute_score_per_patch = [
                5
                if patches[0, i] > mean + 2 * std else 4
                if patches[0, i] > mean + std else 3
                if patches[0, i] > mean else 2
                if patches[0, i] > mean - std else 1
                if patches[0, i] > mean - 2 * std else 0
                for i in range(patches.shape[1])
            ]

            # print(attribute_score_per_patch)

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
