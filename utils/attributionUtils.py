import torch
import numpy as np
from attribution.attention_rollout import AttentionRollout


def process_attribution(attributions, factor):

    norm_cls = prepare_attributions(attributions)

    num_layers = len(attributions)

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

    globenc = AttentionRollout().compute_flows([norm_nenc], disable_tqdm=True, output_hidden_states=True)[0]
    globenc = np.array(globenc)

    norm_cls = globenc[:, 0, :]
    norm_cls = np.flip(norm_cls, axis=0)
    row_sums = norm_cls.max(axis=1)
    norm_cls = norm_cls / row_sums[:, np.newaxis]

    return norm_cls
