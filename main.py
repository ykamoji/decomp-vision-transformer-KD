from fineTuning import fine_tuning
from distillation import run_distillation
from visualisation import visualize
from utils.argUtils import CustomObject, get_yaml_loader
import yaml
import json
loader = yaml.SafeLoader


def start(configPath):
    with open(configPath, 'r') as file:
        config = yaml.load(file, get_yaml_loader())

    x = json.dumps(config)
    Args = json.loads(x, object_hook=lambda d: CustomObject(**d))

    if Args.FineTuning.Action:
        fine_tuning(Args)

    if Args.Distillation.Action:
        run_distillation(Args)

    if Args.Visualization.Action:
        visualize(Args)


if __name__ == '__main__':
    start('config.yaml')

