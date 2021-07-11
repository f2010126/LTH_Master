init_mask = {"linear": 0, "conv": 0, "last": 0}
conv2_prune = {"linear": 0.2, "conv": 0.1, "last": 0.1}
conv4_prune = {"linear": 0.2, "conv": 0.15, "last": 0.1}
lenet_prune = {"linear": 0.2, "conv": 0.2, "last": 0.1}
conv2_lr = 2e-4
default_plot_es = {'title': "Title here",
                   'x_label': "Weights remaining",
                   'y_label': "Early Stop Epoch",
                   'baseline': "full_es",
                   'x_val': 'rem_weight',
                   'y_val': ['pruned_es', 'rand_es'],
                   'y_max': 0,
                   'y_min': 'rand_es'}

default_plot_acc = {'title': "Title here",
                    'x_label': "Weights remaining",
                    'y_label': "Validation Accuracy",
                    'baseline': "val_score",
                    'x_val': 'rem_weight',
                    'y_val': ['val_score', 'rand_init'],
                    'y_max': 100,
                    'y_min': 'rand_init'}
