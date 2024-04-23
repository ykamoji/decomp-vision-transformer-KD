import torch
import numpy as np
from attribution.attention_rollout import AttentionRollout


def process_features(features, factor, featureType):

    if featureType == 'Attribution':
        norm_cls = prepare_attributions(features)
    else:
        norm_cls = prepare_attentions(features)

    num_layers = len(features)

    # print(norm_cls.shape)

    cls = norm_cls[:, 0]
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


def prepare_attributions(attributions):
    num_layers = len(attributions)
    norm_nenc = torch.stack([attributions[i][4] for i in range(num_layers)]).detach().squeeze().cpu().numpy()
    return process_common(norm_nenc)


def prepare_attentions(attentions):
    num_layers = len(attentions)
    attn = torch.stack([attentions[i] for i in range(num_layers)]).detach().squeeze().cpu().numpy()
    norm_attn = attn.mean(axis=1)
    return process_common(norm_attn)


def process_common(attn):
    attn = AttentionRollout().compute_flows([attn], disable_tqdm=True, output_hidden_states=True)[0]
    attn_cls = attn[:, 0, :]
    attn_cls = np.flip(attn_cls, axis=0)
    row_sums = attn_cls.max(axis=1)
    attn_cls = attn_cls / row_sums[:, np.newaxis]
    return attn_cls