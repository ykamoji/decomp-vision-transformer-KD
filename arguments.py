import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Common script', add_help=False)

    parser.add_argument('--model', default='google/vit-base-patch16-224-in21k', type=str, metavar='MODEL',
                        help='Pretrained model to finetune')

    parser.add_argument('--model_dir', default='model/', help='Directory to save the pretrained model')
    parser.add_argument('--results', default='results/',
                        help='Root directory to save models, metrics, logs')

    parser.add_argument('--fine_tuned_dir', default='tuned-model/', help='Directory to save the fine tuned model')

    parser.add_argument('--metrics', default="accuracy", help='Metric(s) to log')
    parser.add_argument('--metrics_dir', default='metrics/', help='Cache directory for storing metrics')

    parser.add_argument('--distilled_dir', default='distilled-model/')

    parser.add_argument('--dataset', default='cifar100', help='Dataset name')
    parser.add_argument('--dataset_labels', default='label', help='Dataset name')
    parser.add_argument('--dataset_dir', default='', help='Dataset directory')
    parser.add_argument('--train', default='', help='Training dataset size')
    parser.add_argument('--test', default='', help='Testing dataset size')

    return parser


def get_fine_tune_args():
    parser = argparse.ArgumentParser('Model fine tuning script', add_help=True, parents=[get_args_parser()])

    parser.add_argument('--train_batch_size', default=32, type=int, help='Fine tuning train batch size')
    parser.add_argument('--val_batch_size', default=32, type=int, help='Fine tuning validation batch size')
    parser.add_argument('--epochs', default=1, type=int, help='Fine tuning epochs')
    parser.add_argument('--lr', default=5e-05, type=float, help='Fine tuning learning rate')
    parser.add_argument('--weight_decay', default=0, type=float, help='Fine tuning weight decay')

    return parser.parse_args()


def get_distillation_args():
    parser = argparse.ArgumentParser('Model distillation training script', add_help=True,
                                     parents=[get_args_parser()])

    parser.add_argument('--student_model', default='facebook/deit-base-distilled-patch16-224',
                        type=str, metavar='MODEL', help='Pretrained model to distill')

    parser.add_argument('--train_batch_size', default=32, type=int, help='Distillation train batch size')
    parser.add_argument('--val_batch_size', default=32, type=int, help='Distillation validation batch size')
    parser.add_argument('--epochs', default=1, type=int, help='Distillation training epochs')
    parser.add_argument('--lr', default=5e-05, type=float, help='Distillation training learning rate')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='Distillation training weight decay')
    parser.add_argument('--distillation_token', default=True, type=bool,
                        help='For pretrained distilled models')
    parser.add_argument('--distillation_type', default='soft', help='To use KL divergence')

    return parser.parse_args()
