IGNORE_KEYS = ['cls_logits', 'distillation_logits', 'hidden_states', 'attentions', 'attributions']


def start_training(Args, trainer, loadCheckpoint, model, output_path, model_output, testing_data):

    with open(output_path + '/training/config.json', 'x', encoding='utf-8') as f:
        f.write(Args.toJSON())

    if loadCheckpoint:
        train_results = trainer.train(ignore_keys_for_eval=IGNORE_KEYS, resume_from_checkpoint=model)
    else:
        train_results = trainer.train(ignore_keys_for_eval=IGNORE_KEYS)

    trainer.save_model(output_dir=output_path + model_output)
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    metrics = trainer.evaluate(testing_data, ignore_keys=IGNORE_KEYS)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)