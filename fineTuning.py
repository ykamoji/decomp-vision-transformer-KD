from process_datasets import build_dataset, build_metrics, collate_fn, collate_ImageNet_fine_tuning_fn
from transformers import Trainer, TrainingArguments
from models_utils import ViTForImageClassification, DeiTForImageClassification
from transformers.training_args import OptimizerNames
from utils.pathUtils import prepare_output_path, get_checkpoint_path
from utils.commonUtils import start_training
import warnings

warnings.filterwarnings('ignore')


def get_fine_tuning_trainer_args(output_path, hyperparameters):
    return TrainingArguments(
        output_dir=output_path + 'training/',
        logging_dir=output_path + 'logs/',
        per_device_train_batch_size=hyperparameters.TrainBatchSize,
        per_device_eval_batch_size=hyperparameters.EvalBatchSize,
        evaluation_strategy="steps",
        num_train_epochs=hyperparameters.Epochs,
        save_steps=hyperparameters.Steps.SaveSteps,
        eval_steps=hyperparameters.Steps.EvalSteps,
        logging_steps=hyperparameters.Steps.LoggingSteps,
        learning_rate=hyperparameters.Lr,
        lr_scheduler_type='cosine',
        warmup_ratio=hyperparameters.WarmUpRatio,
        weight_decay=hyperparameters.WeightDecay,
        save_total_limit=2,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        optim=OptimizerNames.ADAMW_HF,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        seed=42,
        gradient_accumulation_steps=hyperparameters.Steps.GradientAccumulation,
    )


def fine_tuning(Args):
    num_labels, training_data, testing_data = build_dataset(True, Args)

    compute_metrics = build_metrics(Args.Common.Metrics)

    output_path = prepare_output_path('FineTuned', Args)

    fine_tune_args = get_fine_tuning_trainer_args(output_path, Args.FineTuning.Hyperparameters)

    model = Args.FineTuning.Model.Name
    if "deit" in model and "distilled" in model:
        classificationMode = DeiTForImageClassification
    else:
        classificationMode = ViTForImageClassification

    if Args.FineTuning.Model.LoadCheckPoint:
        model = get_checkpoint_path('FineTuned', Args)

    pretrained_model = classificationMode.from_pretrained(model, num_labels=num_labels,
                                                                 cache_dir=Args.FineTuning.Model.CachePath,
                                                                 ignore_mismatched_sizes=True)

    fine_tune_trainer = Trainer(
        model=pretrained_model,
        args=fine_tune_args,
        data_collator=collate_fn if Args.Common.DataSet.Name != "imageNet" else collate_ImageNet_fine_tuning_fn,
        compute_metrics=compute_metrics,
        train_dataset=training_data,
        eval_dataset=testing_data,
    )

    start_training(Args, fine_tune_trainer, Args.FineTuning.Model.LoadCheckPoint, model, output_path,
                   Args.FineTuning.Model.OutputPath, testing_data)
