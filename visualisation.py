from models_utils import ViTForImageClassification
from transformers import ViTImageProcessor
from transformers.image_transforms import resize
from transformers.image_utils import  PILImageResampling
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as pat
import torch
import matplotlib.pyplot as plt
import glob
import numpy as np
from PIL import Image
import pandas as pd
import seaborn as sns
from Attribution.attention_rollout import AttentionRollout
import warnings

warnings.filterwarnings('ignore')

def visualize(Args):

    images = glob.glob("images/*.JPEG")
    Device = Args.Visualization.Model.Device

    # model_path = get_model_path('FineTuned', Args)
    model = ViTForImageClassification.from_pretrained(Args.Visualization.Model.Name,
                                                      cache_dir=Args.Visualization.Model.CachePath)
    model.eval()
    model.to(Device)
    processor = ViTImageProcessor.from_pretrained(Args.Visualization.Model.Name,
                                                  cache_dir=Args.Visualization.Model.CachePath)

    for im in images:
        image = Image.open(im)
        factor = 8

        img_resized = resize(np.array(image), size=(224, 224), resample=PILImageResampling.BILINEAR)
        grid_color = [0, 0, 0]
        grid_size = 224 // factor
        # img_resized_grid = img_resized.copy()
        # img_resized_grid[:, ::grid_size, :] = grid_color
        # img_resized_grid[::grid_size, :, :] = grid_color
        #
        # plt.imshow(img_resized_grid)
        # plt.show()

        inputs = processor(images=image, return_tensors="pt")
        inputs.to(Device)
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True, output_norms=True,
                              output_globenc=True)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        print(f"Actual: {im.split('_')[1].split('.')[0].ljust(10, ' ')} Predicted: {model.config.id2label[predicted_class_idx]}")

        num_layers = len(outputs.attentions)
        norm_nenc = torch.stack([outputs.attributions[i][4] for i in range(num_layers)]).detach().squeeze().cpu().numpy()

        globenc = AttentionRollout().compute_flows([norm_nenc], disable_tqdm=True, output_hidden_states=True)[0]
        globenc = np.array(globenc)

        norm_cls = globenc[:, 0, :]
        norm_cls = np.flip(norm_cls, axis=0)
        row_sums = norm_cls.max(axis=1)
        norm_cls = norm_cls / row_sums[:, np.newaxis]

        # print(norm_cls.shape)

        cls = norm_cls[: ,0]
        # print(cls.shape)

        others = norm_cls[:, 1:]
        # print(others.shape)
        all_patches = []
        step = 196 // (factor**2)
        for layer in range(num_layers):
            patches = []
            for patch in range(0, 196, step):
                patch_attribution = sum(others[layer, patch:patch+step])
                patches.append(patch_attribution)

            all_patches.append(patches)

        all_patches = np.array(all_patches)
        all_patches = all_patches[:,:-1]
        # print(cls)

        # plt.figure(figsize=(7, 7))
        # df = pd.DataFrame(all_patches, columns=np.arange(1, all_patches.shape[1]+1), index=range(len(all_patches), 0, -1))
        # ax = sns.heatmap(df, cmap="Reds", square=True)
        # bottom, top = ax.get_ylim()
        # ax.set_ylim(bottom + 0.5, top - 0.5)
        # plt.title("GlobEnc", fontsize=16)
        # plt.ylabel("Layer", fontsize=16)
        # plt.xticks(rotation = 90, fontsize=16)
        # plt.yticks(fontsize=13)
        # plt.gcf().subplots_adjust(bottom=0.2)
        #
        # plt.show()

        # print(all_patches[0])

        mean = np.mean(all_patches[0])

        attribute_score_per_patch = [1 if all_patches[0, i] > mean else 0 for i in range(all_patches.shape[1])]

        # print(attribute_score_per_patch)

        img_resized_final = img_resized.copy()
        fig, ax = plt.subplots()
        ax.imshow(img_resized_final)
        x, y = np.meshgrid(np.arange(0, factor), np.arange(0, factor), indexing='ij')

        for i in range(factor):
            for j in range(factor):
                index = x[i,j]*factor + y[i,j]
                if attribute_score_per_patch[index] == 1:
                    rect = pat.Rectangle((y[i,j] * grid_size, x[i,j] * grid_size), grid_size - 1, grid_size -1, linewidth=1,
                                         edgecolor='r', facecolor='none')

                    ax.add_patch(rect)

        plt.show()








