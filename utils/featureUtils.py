import torch
import random
import numpy as np
import matplotlib.patches as pat
import matplotlib.pyplot as plt
from attribution.attention_rollout import AttentionRollout


def process_features(features, factor, featureType):

    if featureType == 'Attribution':
        norm_cls = prepare_attributions(features)
    elif featureType == 'Attention':
        norm_cls = prepare_attentions(features)
    elif featureType == 'ATS':
        norm_cls = prepare_ats(features)

    num_layers = len(features)

    # print(norm_cls.shape)
    # cls = norm_cls[:, 0]
    # print(cls.shape)

    if featureType == 'ATS':
        others = norm_cls
    else:
        others = norm_cls[:, 1:]
    # print(others.shape)
    patches = []
    step = 196 // (factor ** 2)
    if step == 0:
        step = 1
    for layer in range(num_layers):
        patches_mini = []
        for patch in range(0, 196, step):
            patch_attribution = sum(others[layer, patch:patch + step])
            patches_mini.append(patch_attribution)
        patches.append(patches_mini)

    patches = np.array(patches)
    # print(patches.shape)
    if step != 1:
        patches = patches[:, :-1]
    # print(cls)

    return patches


def prepare_attributions(attributions):
    num_layers = len(attributions)
    if type(attributions[0]) is tuple:
        norm_nenc = torch.stack([attributions[i][4] for i in range(num_layers)]).detach().squeeze().cpu().numpy()
    else:
        norm_nenc = attributions.detach().cpu().numpy()
    return process_common(norm_nenc)


def prepare_attentions(attentions):
    num_layers = len(attentions)
    attn = torch.stack([attentions[i] for i in range(num_layers)]).detach().squeeze().cpu().numpy()
    norm_attn = attn.mean(axis=1)
    return process_common(norm_attn)


def prepare_ats(adapative):
    num_layers = len(adapative)
    ats = torch.stack([adapative[i] for i in range(num_layers)]).detach().squeeze().cpu().numpy()
    ats = np.flip(ats, axis=0)
    return ats


def process_common(attn):
    attn = AttentionRollout().compute_flows([attn], disable_tqdm=True, output_hidden_states=True)[0]
    attn_cls = attn[:, 0, :]
    attn_cls = np.flip(attn_cls, axis=0)
    row_sums = attn_cls.max(axis=1)
    attn_cls = attn_cls / row_sums[:, np.newaxis]
    return attn_cls


def feature_score(patches):
    mean = np.mean(patches[0])
    std = np.std(patches[0])
    score_per_patch = [
        5
        if patches[0, i] > mean + 2 * std else 4
        if patches[0, i] > mean + std else 3
        if patches[0, i] > mean else 2
        if patches[0, i] > mean - std else 1
        if patches[0, i] > mean - 2 * std else 0
        for i in range(patches.shape[1])
    ]
    return score_per_patch


def plot_feature_scores(attribute_score_per_patch, ax, factor, feature_type, grid_size, img_resized_feature,
                        threshold_score):
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
                ax.imshow(img_resized_feature)
                ax.set_title(f"{feature_type}")
                ax.axis('off')


def mask_image(image_tensor, type, mask_perc, scores=None, threshold_score=2):
    if type == 'random':
        image_masked = torch.clone(image_tensor)
        _, height, width = image_tensor.shape

        mask_pixel_count = int(height * width * mask_perc / 100)
        indices = random.sample(range(height * width), mask_pixel_count)

        for index in indices:
            row = index // width
            col = index % width
            image_masked[:,row, col] = torch.tensor([0, 0, 0])

        return image_masked

    else:
        image_masked = torch.clone(image_tensor)
        _, height, width = image_tensor.shape
        factor = 14
        grid_size = 224 // factor
        x, y = np.meshgrid(np.arange(0, factor), np.arange(0, factor), indexing='ij')
        # fig, ax = plt.subplots(1,1)
        # plot_feature_scores(scores, ax, factor, type, grid_size, image_masked.permute((1,2,0)).numpy(), threshold_score)
        # plt.show()

        pixels_to_mask = []
        for i in range(14):
            for j in range(14):
                index = x[i, j] * 14 + y[i, j]
                score = scores[index]
                if score >= threshold_score:
                    mask_x, mask_y = np.meshgrid(np.arange(x[i, j] * grid_size, (x[i, j] + 1)* grid_size),
                                np.arange(y[i, j] * grid_size, (y[i, j] + 1)* grid_size),
                                indexing='ij')

                    for xi in range(grid_size):
                        for yj in range(grid_size):
                            pixels_to_mask.append(((mask_x[xi, yj], mask_y[xi, yj]), score))

        pixels_to_mask_sorted = sorted(pixels_to_mask, key=lambda pixel: pixel[1])

        mask_pixel_count = int(height * width * mask_perc / 100)
        if mask_pixel_count > len(pixels_to_mask):
            mask_pixel_count = len(pixels_to_mask)

        pixels_to_mask_perc = [(x,y) for ((x,y),score) in pixels_to_mask_sorted[:mask_pixel_count]]

        for x, y in pixels_to_mask_perc:
            image_masked[:,x,y] = torch.tensor([0, 0, 0])

        # plt.imshow(image_masked.permute((1,2,0)).numpy())
        # plt.show()

        return image_masked




