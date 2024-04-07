import os

def prepare_output_path(step, args):
    root = args.results + step + '/'
    output_path = check_and_update_existing_path(root, args.model)
    output_path += '/'
    return output_path


def check_and_update_existing_path(root, model):
    run_index = 0
    model_path = root + model.split('/')[-1]
    if os.path.exists(model_path):
        current_index = max([int(run.split("run_")[-1]) for run in os.listdir(model_path)])
        run_index = current_index + 1
    path = model_path + f"/run_{run_index}"
    return path


