from process_datasets import build_dataset, build_metrics, collate_fn, collate_imageNet_fn
from transformers import TrainingArguments, ViTConfig
from models_utils import ViTForImageClassification, DeiTForImageClassificationWithTeacher
from transformers.training_args import OptimizerNames
from loss import DistillationTrainer
from torch.utils.tensorboard import SummaryWriter
from utils.pathUtils import prepare_output_path, get_model_path, get_checkpoint_path
from utils.commonUtils import start_training
import warnings

warnings.filterwarnings('ignore')


def get_distillation_training_args(output_path, hyperparameters):
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
        label_names=['labels'],
    )


def run_distillation(Args):
    try:
        if Args.Distillation.Model.UseLocal:
            fine_tuned_model_path = get_model_path('FineTuned', Args)
        else:
            fine_tuned_model_path = ''
        teacher_model = ViTForImageClassification.from_pretrained(fine_tuned_model_path)
    except Exception as e:
        print(f"Warning: {e}. Using huggingface pretrained model.")
        teacher_model = ViTForImageClassification.from_pretrained(Args.Distillation.Model.Name,
                                                                  cache_dir=Args.Distillation.Model.CachePath)
    # print(teacher_model)

    _, training_data, testing_data = build_dataset(True, Args)

    compute_metrics = build_metrics(Args.Common.Metrics)

    if Args.Distillation.UseDistTokens:
        classificationMode = DeiTForImageClassificationWithTeacher
    else:
        classificationMode = ViTForImageClassification

    model = Args.Distillation.StudentModel.Name

    if Args.Distillation.StudentModel.LoadCheckPoint:
        model = get_checkpoint_path('Distilled', Args)

    if Args.Distillation.RandomWeights:

        student_config = ViTConfig.from_pretrained(model, num_labels=teacher_model.config.num_labels,
                                                   cache_dir=Args.Distillation.StudentModel.CachePath,
                                                   ignore_mismatched_sizes=True)

        student_model = classificationMode._from_config(config=student_config)

    else:

        student_model = classificationMode.from_pretrained(model, num_labels=teacher_model.config.num_labels,
                                                           cache_dir=Args.Distillation.StudentModel.CachePath,
                                                           ignore_mismatched_sizes=True)

    output_path = prepare_output_path('Distilled', Args)

    distillation_args = get_distillation_training_args(output_path, Args.Distillation.Hyperparameters)

    writer = SummaryWriter(distillation_args.logging_dir + 'loss/', max_queue=10, flush_secs=10)

    distillation_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        args=distillation_args,
        data_collator=collate_fn if Args.Common.DataSet.Name != "imageNet" else collate_imageNet_fn,
        compute_metrics=compute_metrics,
        train_dataset=training_data,
        eval_dataset=testing_data,
        temperature=Args.Distillation.Hyperparameters.KD.Temperature,
        alpha=Args.Distillation.Hyperparameters.KD.Alpha,
        writer=writer,
        configArgs=Args
    )

    start_training(Args, distillation_trainer, Args.Distillation.StudentModel.LoadCheckPoint, model, output_path,
                   Args.Distillation.StudentModel.OutputPath, testing_data)

    writer.close()
