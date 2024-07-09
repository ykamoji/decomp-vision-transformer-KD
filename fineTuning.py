from process_datasets import build_dataset, build_metrics, collate_fn
from transformers import Trainer, TrainingArguments
from models_utils import ViTForImageClassification, DeiTForImageClassification
from transformers.training_args import OptimizerNames
from utils.pathUtils import prepare_output_path
import warnings

warnings.filterwarnings('ignore')

IGNORE_KEYS = ['cls_logits', 'distillation_logits', 'hidden_states', 'attentions', 'attributions']


def get_fine_tuning_trainer_args(output_path, hyperparameters):
    return TrainingArguments(
        output_dir=output_path + 'training/',
        logging_dir=output_path + 'logs/',
        per_device_train_batch_size=hyperparameters.TrainBatchSize,
        per_device_eval_batch_size=hyperparameters.EvalBatchSize,
        evaluation_strategy="steps",
        num_train_epochs=hyperparameters.Epochs,
        save_steps=20,
        eval_steps=20,
        logging_steps=10,
        learning_rate=hyperparameters.Lr,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        warmup_steps=100,
        weight_decay=hyperparameters.WeightDecay,
        save_total_limit=2,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        optim=OptimizerNames.ADAMW_HF,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        seed=42,
        gradient_accumulation_steps=1,
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

    pretrained_model = classificationMode.from_pretrained(model, num_labels=num_labels,
                                                                 cache_dir=Args.FineTuning.Model.CachePath,
                                                                 ignore_mismatched_sizes=True)

    fine_tune_trainer = Trainer(
        model=pretrained_model,
        args=fine_tune_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=training_data,
        eval_dataset=testing_data,
    )

    train_results = fine_tune_trainer.train(ignore_keys_for_eval=IGNORE_KEYS)

    fine_tune_trainer.save_model(output_dir=output_path + Args.FineTuning.Model.OutputPath)
    fine_tune_trainer.log_metrics("train", train_results.metrics)
    fine_tune_trainer.save_metrics("train", train_results.metrics)
    fine_tune_trainer.save_state()

    metrics = fine_tune_trainer.evaluate(testing_data, ignore_keys=IGNORE_KEYS)
    fine_tune_trainer.log_metrics("eval", metrics)
    fine_tune_trainer.save_metrics("eval", metrics)

    with open(output_path + '/training/'+'config.json', 'x', encoding='utf-8') as f:
        f.write(Args.toJSON())
