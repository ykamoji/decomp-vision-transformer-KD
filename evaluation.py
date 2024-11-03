from process_datasets import build_dataset, build_metrics, collate_fn, collate_ImageNet_fine_tuning_fn
from transformers import Trainer, TrainingArguments
from models_utils import ViTForImageClassification, DeiTForImageClassification
from utils.pathUtils import prepare_output_path, get_model_path, get_checkpoint_path
from utils.commonUtils import start_evaluation, save_config
import warnings

warnings.filterwarnings('ignore')


def get_evaluation_args(output_path, hyperparameters):
    return TrainingArguments(
        output_dir=output_path,
        per_device_eval_batch_size=hyperparameters.EvalBatchSize,
        remove_unused_columns=False,
        seed=42,
        do_train=False,
        do_eval=True,
        report_to=[]
    )


def evaluate(Args):

    _, training_data, testing_data = build_dataset(False, Args)

    compute_metrics = build_metrics(Args.Common.Metrics)

    output_path = prepare_output_path('Evaluation', Args)

    eval_args = get_evaluation_args(output_path, Args.Evaluate.Hyperparameters)

    model = Args.Evaluate.Model.Name
    if "deit" in model and "distilled" in model:
        classificationMode = DeiTForImageClassification
    else:
        classificationMode = ViTForImageClassification

    try:
        if Args.Evaluate.Model.LoadCheckPoint:
            model_path = get_checkpoint_path(Args.Evaluate.Model.Type, Args)
        elif Args.Distillation.Model.UseLocal:
            model_path = get_model_path(Args.Evaluate.Model.Type, Args)
        else:
            model_path = ''
        model = classificationMode.from_pretrained(model_path)
    except Exception as e:
        print(f"Warning: {e}. Using huggingface pretrained model.")
        model = classificationMode.from_pretrained(Args.Evaluate.Model.Name,
                                                   cache_dir=Args.Evaluate.Model.CachePath)

    eval_trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=collate_fn if Args.Common.DataSet.Name != "imageNet" else collate_ImageNet_fine_tuning_fn,
        compute_metrics=compute_metrics,
        train_dataset=training_data,
        eval_dataset=testing_data,
    )

    save_config(output_path + 'config.yaml', Args.Evaluate)
    start_evaluation(eval_trainer, testing_data)
