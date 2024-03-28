import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Common script', add_help=False)

    parser.add_argument('--model', default='google/vit-base-patch16-224-in21k', type=str, metavar='MODEL',
                        help='Name of model to finetune')
    parser.add_argument('--model_dir', default='model/')
    parser.add_argument('--results', default='results/')

    parser.add_argument('--fine_tuned_dir', default='fine-tuned-models/')

    parser.add_argument('--student_model', default='facebook/deit-base-distilled-patch16-224',
                        type=str, metavar='MODEL', help='Name of model to distill')
    parser.add_argument('--distilled_dir', default='distilled-models/')

    parser.add_argument('--metrics', default="accuracy")
    parser.add_argument('--metrics_dir', default='metrics/')

    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--dataset_dir', default='')
    parser.add_argument('--train', default='1%')
    parser.add_argument('--test', default='1%')

    return parser


def get_fine_tune_args():
    parser = argparse.ArgumentParser('Model fine tuning script', add_help=True, parents=[get_args_parser()])

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)

    return parser.parse_args()


def get_distillation_args():
    parser = argparse.ArgumentParser('Model distillation training script', add_help=True, parents=[get_args_parser()])

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)

    return parser.parse_args()
