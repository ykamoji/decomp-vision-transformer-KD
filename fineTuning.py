from arguments import get_fine_tune_args
from process_datasets import build_dataset, build_metrics, collate_fn
from transformers import Trainer, TrainingArguments, ViTForImageClassification


def get_fine_tuning_trainer_args(args):

    output_path = args.results + 'fine-tuning-' + args.model.split('/')[-1] + '/'

    return TrainingArguments(
        output_dir=output_path,
        logging_dir=output_path + 'logs/',
        per_device_train_batch_size=args.batch_size,
        evaluation_strategy="steps",
        num_train_epochs=args.epochs,
        save_steps=100,
        eval_steps=50,
        logging_steps=10,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True
    )


def fine_tuning(args):
    print(args)

    num_labels, training_data, testing_data = build_dataset(True, args)

    compute_metrics = build_metrics(args)

    fine_tune_args = get_fine_tuning_trainer_args(args)

    pretrained_model = ViTForImageClassification.from_pretrained(args.model,
                                                                 num_labels=num_labels,
                                                                 cache_dir=args.model_dir,
                                                                 ignore_mismatched_sizes=True)

    fine_tune_trainer = Trainer(
        model=pretrained_model,
        args=fine_tune_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=training_data,
        eval_dataset=testing_data,
    )

    train_results = fine_tune_trainer.train()

    fine_tune_trainer.save_model(output_dir=args.results + args.fine_tuned_dir + args.model)
    fine_tune_trainer.log_metrics("train", train_results.metrics)
    fine_tune_trainer.save_metrics("train", train_results.metrics)
    fine_tune_trainer.save_state()

    metrics = fine_tune_trainer.evaluate(testing_data)
    fine_tune_trainer.log_metrics("eval", metrics)
    fine_tune_trainer.save_metrics("eval", metrics)


if __name__ == '__main__':
    fine_tuning(get_fine_tune_args())
