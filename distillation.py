from process_datasets import build_dataset, build_metrics, collate_fn
from transformers import TrainingArguments
from models_utils import ViTForImageClassification, DeiTForImageClassification, DeiTForImageClassificationWithTeacher
from transformers.training_args import OptimizerNames
from loss import DistillationTrainer
from utils.pathUtils import prepare_output_path, get_model_path
import warnings

warnings.filterwarnings('ignore')

IGNORE_KEYS = ['cls_logits', 'distillation_logits', 'hidden_states', 'attentions']


def get_distillation_training_args(output_path, hyperparameters):

    return TrainingArguments(
        output_dir=output_path + 'training/',
        logging_dir=output_path + 'logs/',
        per_device_train_batch_size=hyperparameters.TrainBatchSize,
        per_device_eval_batch_size=hyperparameters.EvalBatchSize,
        evaluation_strategy="steps",
        num_train_epochs=hyperparameters.Epochs,
        save_steps=30,
        eval_steps=30,
        logging_steps=10,
        learning_rate=hyperparameters.Lr,
        warmup_ratio=0.1,
        warmup_steps=1,
        weight_decay=hyperparameters.WeightDecay,
        save_total_limit=2,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        optim=OptimizerNames.ADAMW_HF,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        seed=42,
        gradient_accumulation_steps=4,
        label_names=['labels'],
    )


def run_distillation(Args):

    fine_tuned_model_path = get_model_path('FineTuned', Args)

    teacher_model = ViTForImageClassification.from_pretrained(fine_tuned_model_path)

    # print(teacher_model)

    ## TODO:: Extend the student model to use linear transformation for the layer at the end.
    ##  refer https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/TinyBERT/transformer/modeling.py#L1119

    num_labels, training_data, testing_data = build_dataset(True, Args, show_details=False)

    compute_metrics = build_metrics(Args.Common.Metrics)

    if Args.Distillation.UseDistTokens:
        classificationMode = DeiTForImageClassificationWithTeacher
    else:
        classificationMode = ViTForImageClassification

    student_model = classificationMode.from_pretrained(Args.student_model,
                                                       num_labels=num_labels,
                                                       cache_dir=Args.model_dir,
                                                       ignore_mismatched_sizes=True)

    # print(classificationMode)

    output_path = prepare_output_path('Distilled', Args)

    distillation_args = get_distillation_training_args(output_path, Args.Distillation.Hyperparameters)

    distillation_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        args=distillation_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=training_data,
        eval_dataset=testing_data,
        temperature=5,
        alpha=0.5,
        distillation_token=Args.distillation_token,
        distillation_type=Args.distillation_type
    )

    train_results = distillation_trainer.train(ignore_keys_for_eval=IGNORE_KEYS)

    distillation_trainer.save_model(output_dir=output_path + Args.distilled_dir)
    distillation_trainer.log_metrics("train", train_results.metrics)
    distillation_trainer.save_metrics("train", train_results.metrics)
    distillation_trainer.save_state()

    metrics = distillation_trainer.evaluate(testing_data, ignore_keys=IGNORE_KEYS)
    distillation_trainer.log_metrics("eval", metrics)
    distillation_trainer.save_metrics("eval", metrics)


