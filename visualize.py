from models_utils import ViTForImageClassification
from arguments import get_args_parser
from transformers import TrainingArguments, Trainer
from process_datasets import build_dataset, collate_fn, build_metrics
from utils.pathUtils import prepare_output_path
import warnings

warnings.filterwarnings('ignore')

def get_visualization_testing_args(output_path):

    return TrainingArguments(
        output_dir=output_path + 'training/',
        logging_dir=output_path + 'logs/',
        remove_unused_columns=False,
        push_to_hub=False,
        seed=42,
        do_eval=True
    )


def visualise_attributes(args):

    test_data = build_dataset(False, args)

    pretrained_model = ViTForImageClassification.from_pretrained(args.model, cache_dir=args.model_dir)

    pretrained_model.eval()

    compute_metrics = build_metrics(args)

    output_path = prepare_output_path('Visualize', args)

    visualization_args = get_visualization_testing_args(output_path)

    visualize_trainer = Trainer(
        model=pretrained_model,
        args=visualization_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        eval_dataset=test_data
    )

    metrics = visualize_trainer.evaluate(test_data)
    visualize_trainer.log_metrics("eval", metrics)
    visualize_trainer.save_metrics("eval", metrics)



if __name__ == "__main__":
    visualise_attributes(get_args_parser().parse_args())