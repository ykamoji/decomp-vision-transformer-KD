from arguments import get_fine_tune_args, get_distillation_args
from fineTuning import fine_tuning
from distillation import run_distillation

if __name__ == '__main__':
    fine_tuning(get_fine_tune_args())
    run_distillation(get_distillation_args())