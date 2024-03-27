import argparse
from process_datasets import build_dataset, collate_fn, build_metrics
from transformers import Trainer, TrainingArguments, ViTForImageClassification

def get_args_parser():
    parser = argparse.ArgumentParser('Model fine tuning script', add_help=False)

    parser.add_argument('--model', default='google/vit-base-patch16-224-in21k', type=str, metavar='MODEL',
                        help='Name of model to finetune')
    parser.add_argument('--model_dir', default='model/')
    parser.add_argument('--results', default='results/')

    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--epochs', default=2, type=int)

    parser.add_argument('--metrics', default="accuracy")
    parser.add_argument('--metrics_dir', default='metrics/')

    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--dataset_dir', default='')
    parser.add_argument('--train', default='5%')
    parser.add_argument('--test', default='5%')

    return parser

def getTraingingArgs(args):

    return TrainingArguments(
        output_dir=args.results,
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=args.epochs,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=1e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True
    )


def main(args):
    print(args)

    num_labels, training_data, testing_data = build_dataset(True, args)

    compute_metrics = build_metrics(args)

    train_args = getTraingingArgs(args)

    pretrained_model = ViTForImageClassification.from_pretrained(args.model,
                                                                 num_labels=num_labels,
                                                                 cache_dir=args.model_dir)

    trainer = Trainer(
        model=pretrained_model,
        args=train_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=training_data,
        eval_dataset=testing_data,
    )

    train_results = trainer.train()

    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    # save the trainer state
    trainer.save_state()

    metrics = trainer.evaluate(testing_data)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model fine tuning script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
