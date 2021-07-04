import json
import pandas as pd
import os
import matplotlib.ticker as mtick
import math
import torch
import LTH_Constants
import matplotlib.pyplot as plt

#TODO: shift inside each class?
def init_weights(m):
    """
        Initialise weights acc the Xavier initialisation and bias set to 0.01
        :param m:
        :return:
        """
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def countRemWeights(model):
    """
    Percetage of weights that remain for training
    :param model:
    :return: % of weights remaining
    """
    total_weights = 0
    rem_weights = 0
    for name, module in model.named_modules():
        if any([isinstance(module, cl) for cl in [torch.nn.Conv2d, torch.nn.Linear]]):
            rem_weights += torch.count_nonzero(module.weight)
            total_weights += sum([param.numel() for param in module.parameters()])
    # return % of non 0 weights
    return rem_weights.item() / total_weights * 100


def save_data(json_dump, name_file="exp_result.json"):
    """
    Saves the results as a json file
    :param json_dump: data to store
    :param name_file: filename
    :return: location of file
    """
    json_path = os.path.join(os.getcwd(), "LTH_Results")
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    fname = os.path.join(json_path, name_file)
    fh = open(fname, "w")
    json.dump(json_dump, fh)
    return fname


def rounddown(x):
    return int(math.floor(x / 10.0)) * 10


def plot_from_json(json_loc):
    """

    :return:
    """
    results_dir = os.path.join(os.getcwd(), "LTH_Results")
    json_loc = "prune_Net2_cifar10_20.json"
    with open(os.path.join(results_dir, json_loc)) as json_file:
        data = json.load(json_file)
        plot_config = LTH_Constants.default_plot_acc
        file_name = json_loc.split('.')[0]
        plot_config['title'] = file_name + "From code"
        plot_config['y_max'] = 100
        plot_graph(data, plot_config, file_name + "from_code.png")


def plot_train_valid(train_data):
    """

    :return:
    """
    df = pd.DataFrame.from_dict(train_data)
    df.set_index('epoch')
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.suptitle('Training with 41% weights')
    ax1.plot(df['epoch'], df["train_loss"], 'rs',
             df['epoch'], df["val_loss"], 'bo', linestyle='dashed')
    ax1.title.set_text('Loss curve')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(["Training", "Validation"], loc="lower left")
    ax2.plot(df['epoch'], df["train_score"], 'rs',
             df['epoch'], df["val_score"], 'bo', linestyle='dashed')
    ax2.title.set_text('Accuracy curve')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend(["Training", "Validation"], loc="lower left")
    fig.show()
    json_path = os.path.join(os.getcwd(), "LTH_Results")
    fig.savefig(os.path.join(json_path, "Net2_cifar10_training_25_41wt.png"))


def plot_graph(graph_data, plot_config, file_at="pruned.png"):
    """
    Plots the graph. need a baseline and x,y points
    :param graph_data: dict of data
    :param plot_config: graph details and columns
    :param file_at: location of plot
    :return: location
    """
    df = pd.DataFrame.from_dict(graph_data['prune_data'])
    df.set_index('rem_weight')
    # (t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    ax = df.plot(x=plot_config['x_val'], y=plot_config['y_val'], marker='o', title=plot_config['title'])
    ax.axhline(y=graph_data[plot_config['baseline']], color='r', linestyle='-', label=plot_config['baseline'])
    ax.annotate(plot_config['baseline'], (0, graph_data[plot_config['baseline']]))
    fmt = '%.0f%%'  # Format you want the ticks, e.g. '40%'
    ax.set_ylim(rounddown(df[plot_config['y_min']].min()), plot_config['y_max'])
    ax.set_xlim(0, 100)
    ax.invert_xaxis()
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    ax.set_xlabel(plot_config['x_label'])
    ax.set_ylabel(plot_config['y_label'])
    fig = ax.get_figure()
    # fig.show()
    json_path = os.path.join(os.getcwd(), "LTH_Results")
    fig.savefig(os.path.join(json_path, file_at))
    print("")
