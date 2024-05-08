import torch
import random
import numpy as np
import matplotlib.patches as pat
import matplotlib.pyplot as plt
from features.attention_rollout import AttentionRollout
from features.plus import compute_plus, compute_skip_plus
from torchvision.transforms import v2


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'

    return torch.device(device)


def process_features(features, factor, featureType, strategies):

    if featureType == 'Attribution':
        norm_cls = prepare_attributions(features, strategies)
    elif featureType == 'Attention':
        norm_cls = prepare_attentions(features, strategies)
    elif featureType == 'ATS':
        norm_cls = prepare_ats(features, strategies)

    num_layers = len(norm_cls)

    # print(norm_cls.shape)
    # cls = norm_cls[:, 0]
    # print(cls.shape)

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


def prepare_attributions(attributions, strategies):
    num_layers = len(attributions)
    if type(attributions[0]) is tuple:
        norm_nenc = torch.stack([attributions[i][4] for i in range(num_layers)]).detach().squeeze().cpu().numpy()
    elif type(attributions[0]) is not tuple:
        norm_nenc = torch.stack([attributions[i] for i in range(num_layers)]).detach().squeeze().cpu().numpy()
    else:
        norm_nenc = attributions.detach().cpu().numpy()
    return process_common(norm_nenc, strategies, type='Attributions')


def prepare_attentions(attentions, strategies):
    num_layers = len(attentions)
    attn = torch.stack([attentions[i] for i in range(num_layers)]).detach().squeeze().cpu().numpy()
    norm_attn = attn.mean(axis=1)
    return process_common(norm_attn, strategies, type='Attentions')


def prepare_ats(adapative, strategies):
    num_layers = len(adapative)
    ats = torch.stack([adapative[i] for i in range(num_layers)]).detach().squeeze().cpu().numpy()
    return process_common(ats, strategies, type='ATS')


def process_common(attn, strategies, type):

    attn_processed = None
    if "rollout" in strategies:
        if type == 'ATS' and attn.ndim == 2:
            attn_rollout = attn
        else:
            attn_rollout = AttentionRollout().compute_flows([attn], disable_tqdm=True, output_hidden_states=True)[0]
        attn_processed = attn_rollout

    if "plus" in strategies:
        attn_plus = compute_plus(attn)
        if attn_processed:
            attn_processed += attn_plus
        else:
            attn_processed = attn_plus

    if "skipplus_first" in strategies or "skipplus_last" in strategies:
        last = "skipplus_last" in strategies
        attn_skipplus = compute_skip_plus(attn, last)
        if attn_processed:
            attn_processed += attn_skipplus
        else:
            attn_processed = attn_skipplus

    if attn_processed.ndim > 2:
        attn_cls = attn_processed[:, 0, :]
        attn_cls = np.flip(attn_cls, axis=0)
        row_sums = attn_cls.max(axis=1)
        attn_cls = attn_cls / row_sums[:, np.newaxis]
    else:
        attn_cls = attn_processed
        attn_cls = np.flip(attn_cls, axis=0)

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


def mask_image(image_tensor, featureType, mask_perc, scores=None, threshold_score=2):
    image_masked = torch.clone(image_tensor)
    factor = 14
    grid_size = 224 // factor
    channel, _, _ = image_tensor.shape
    mask_pixel_count = int(factor ** 2 * mask_perc / 100)
    if featureType == 'random':
        indices = random.sample(range(factor**2), mask_pixel_count)

        for index in indices:
            row = index // factor
            col = index % factor
            image_masked[:,row*grid_size:(row+1)*grid_size, col*grid_size:(col+1)*grid_size] \
                = torch.rand((channel, grid_size, grid_size))

        return image_masked

    else:
        x, y = np.meshgrid(np.arange(0, factor), np.arange(0, factor), indexing='ij')
        # fig, ax = plt.subplots(1,1)
        # plot_feature_scores(scores, ax, factor, type, grid_size, image_masked.permute((1,2,0)).numpy(), threshold_score)
        # plt.show()

        pixels_to_mask = []
        for i in range(factor):
            for j in range(factor):
                index = x[i, j] * factor + y[i, j]
                score = scores[index]
                if score >= threshold_score:
                    pixels_to_mask.append(((x[i, j], y[i, j]), -score))

        pixels_to_mask_sorted = sorted(pixels_to_mask, key=lambda pixel: pixel[1])

        pixels_to_mask_perc = [(x,y) for ((x,y),score) in pixels_to_mask_sorted[:mask_pixel_count]]

        for x, y in pixels_to_mask_perc:
            image_masked[:,x*grid_size:(x+1)*grid_size,y*grid_size:(y+1)*grid_size] \
                = torch.rand((channel, grid_size,grid_size))

        return image_masked


def show_masked_images(image, label, featureType, scores, mask_percs, Args):
    resize = v2.Resize(size=(224, 224))
    original_image = torch.tensor(image, device=get_device()).permute((2,0,1))
    original_image = resize(original_image)

    fig, ax = plt.subplots(1, len(mask_percs), figsize=(30, 8))
    for index, mask_perc in enumerate(mask_percs):
        original_image = mask_image(original_image, featureType=featureType, mask_perc=mask_perc, scores=scores,
                                threshold_score=Args.Visualization.Plot.ThresholdScore)
        ax[index].imshow(original_image.permute((1, 2, 0)).cpu().numpy())
        ax[index].axis('off')
    # plt.title(featureType)
    if Args.Visualization.Masking.SaveMasking:
        plt.savefig(f'temp/masking_{label}_{featureType}')
    if  Args.Visualization.Masking.ShowMasking:
        plt.show()

