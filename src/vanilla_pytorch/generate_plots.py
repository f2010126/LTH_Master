from utils import plot_graph
import argparse
import json
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LTH Experiments')
    parser.add_argument('--data-path',
                        help='path to data file',
                        type=str, required=True)
    args = parser.parse_args()

    assert os.path.exists(args.data_path), \
        "File {} does not exits,\nPlease make sure file is accessible from here!".format(args.data_path)
    with open(args.data_path) as f:
        run_data = json.load(f)
    file_name = args.data_path.split(os.path.sep)[-1]
    epochs = max([y for x in run_data['prune_data'] for y in [x['rand_es'], x['pruned_es']]])
    plot = {'title': file_name.split('.')[0],
            'x_label': "Weights remaining",
            'y_label': "Early Stop Epoch",
            'baseline': "full_es",
            'x_val': 'rem_weight',
            'y_val': ['pruned_es', 'rand_es'],
            'y_max': epochs + 2,
            'y_min': 'rand_es'}
    plot_graph(run_data, plot, file_at=file_name + "_es.png", save_figure=False)
    plot = {'title': file_name.split('.')[0],
            'x_label': "Weights remaining",
            'y_label': "Validation Accuracy",
            'baseline': "val_score",
            'x_val': 'rem_weight',
            'y_val': ['val_score', 'rand_init'],
            'y_max': 100,
            'y_min': 'rand_init'}
    plot_graph(run_data, plot, file_at=file_name + ".png", save_figure=False)
    plt.show()
