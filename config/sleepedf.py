params = {
    # Train
    "n_epochs": 100,
    "learning_rate": 1e-4,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-8,
    "clip_grad_value": 5.0,
    "evaluate_span": 50,
    "checkpoint_span": 50,

    # Early-stopping
    "no_improve_epochs": 10,

    # Model
    "n_rnn_layers": 1,
    "n_rnn_units": 128,
    "sampling_rate": 100.0,
    "input_size": 3000,
    "n_classes": 5,
    "l2_weight_decay": 1e-3,

    # Dataset
    "dataset": "sleepedf",
    "data_dir": "./data/sleepedf/sleep-cassette/eeg_fpz_cz",
    "n_folds": 20,
    "n_subjects": 20,
}

train = params.copy()
train.update({
    "seq_length": 20,
    "batch_size": 15,
})

predict = params.copy()
predict.update({
    "batch_size": 1,
    "seq_length": 1,
})