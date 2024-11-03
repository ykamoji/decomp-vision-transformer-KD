import yaml
import re

IGNORE_KEYS = ['cls_logits', 'distillation_logits', 'hidden_states', 'attentions', 'attributions']


def represent_bool(self, data):
    if data:
        return self.represent_scalar('tag:yaml.org,2002:bool', 'True')
    return self.represent_scalar('tag:yaml.org,2002:bool', 'False')


yaml.add_representer(bool, represent_bool)


def save_config(output_path, Args):
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml_data = yaml.dump(Args, sort_keys=False,  default_flow_style=False)
        cleaned_yaml = re.sub(r'(\s*!!.*$)|(^!!.*$)', '', yaml_data, flags=re.MULTILINE)
        f.write(cleaned_yaml)


def start_training(Args, trainer, loadCheckpoint, model, output_path, model_output, testing_data):

    save_config(output_path + '/training/config.yaml', Args)

    if loadCheckpoint:
        train_results = trainer.train(ignore_keys_for_eval=IGNORE_KEYS, resume_from_checkpoint=model)
    else:
        train_results = trainer.train(ignore_keys_for_eval=IGNORE_KEYS)

    trainer.save_model(output_dir=output_path + model_output)
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    start_evaluation(trainer, testing_data)


def start_evaluation(trainer, testing_data):
    metrics = trainer.evaluate(testing_data, ignore_keys=IGNORE_KEYS)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

