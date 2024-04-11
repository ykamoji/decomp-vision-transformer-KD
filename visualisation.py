from models_utils import ViTForImageClassification
import torch
from process_datasets import build_dataset
from utils.pathUtils import get_model_path
import warnings

warnings.filterwarnings('ignore')

def visualize(Args):

    test_data = build_dataset(False, Args)

    model_path = get_model_path('FineTuned', Args)

    pretrained_model = ViTForImageClassification.from_pretrained(model_path)

    Device = Args.Visualization.Model.Device

    pretrained_model.to(Device)
    pretrained_model.eval()

    batch = next(test_data.iter(batch_size=int(Args.Common.DataSet.Test))).data
    input = {'pixel_values': batch['pixel_values'].to(Device), 'labels': torch.tensor(batch['label']).to(Device)}

    output = pretrained_model(**input, output_attentions=True, output_hidden_states=True)

    pred = output.logits.argmax(-1).item()
    print("\n\nGround truth class: ", batch['label'])
    print("\nPredicted class:", pretrained_model.config.id2label[pred], "\n\n")

    print(len((output.hidden_states)))
    print(output.hidden_states[0].shape)
    print(len((output.attentions)))
    print(output.attentions[0].shape)

