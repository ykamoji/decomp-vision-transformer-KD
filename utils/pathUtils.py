import os


def prepare_output_path(step, Args):
    root = Args.Common.Results + step + '/'
    model_name = ''
    if step == 'FineTuned' or step == 'Visualize':
        model_name = Args.FineTuning.Model.Name
    elif step == 'Distilled':
        model_name = Args.Distillation.StudentModel.Name

    output_path = check_and_update_model_path(root, Args.Common.DataSet.Name, model_name)
    output_path += '/'
    return output_path


def check_and_update_model_path(root, dataset, model):
    if not model or not model.strip():
        raise Exception("Model name cannot be empty!")
    run_index = 0
    model_path = root + model.split('/')[-1] + "/" + dataset
    if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
        current_index = max([int(run.split("run_")[-1]) for run in os.listdir(model_path)])
        run_index = current_index + 1
    path = model_path + f"/run_{run_index}"
    return path


def check_model_path(root, model, dataset, index):
    model_path = root + model.split('/')[-1] + '/' + dataset
    if os.path.exists(model_path):
        if index == -1:
            latest_index = max([int(run.split("run_")[-1]) for run in os.listdir(model_path)])
        else:
            latest_index = index
    else:
        raise Exception("Fine tuned model not found !")
    path = model_path + f"/run_{latest_index}"
    return path


def get_model_path(step, Args):
    root = Args.Common.Results + step + '/'
    model_path = check_model_path(root, Args.Distillation.Model.Name, Args.Common.DataSet.Name,
                                  Args.Distillation.Model.Index)
    model_path += '/' + Args.Distillation.Model.OutputPath
    return model_path


