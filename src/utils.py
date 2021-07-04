import json
import pandas as pd
import os
import matplotlib.ticker as mtick
import math
import torch


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
    ax.axhline(y=graph_data[plot_config['baseline']], color='r', linestyle='-', label= plot_config['baseline'])
    ax.annotate(plot_config['baseline'], (0, graph_data[plot_config['baseline']]))
    fmt = '%.0f%%'  # Format you want the ticks, e.g. '40%'
    ax.set_ylim(rounddown(df[plot_config['y_min']].min()), plot_config['y_max'])
    ax.set_xlim(0, 100)
    ax.invert_xaxis()
    ax.semilogx()
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    ax.set_xlabel(plot_config['x_label'])
    ax.set_ylabel(plot_config['y_label'])
    fig = ax.get_figure()
    # fig.show()
    json_path = os.path.join(os.getcwd(), "LTH_Results")
    fig.savefig(os.path.join(json_path, file_at))
    print("")
