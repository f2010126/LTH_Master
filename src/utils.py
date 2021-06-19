import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mtick
import math


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


def plot_graph(graph_data, file_at="pruned.png"):
    df = pd.DataFrame.from_dict(graph_data['prune_data'])
    df.set_index('rem_weight')
    # (t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    ax = df.plot(x="rem_weight", y=["val_score", "rand_init"], title="MNSIT")
    ax.axhline(y=graph_data['baseline'], color='r', linestyle='-')
    fmt = '%.0f%%'  # Format you want the ticks, e.g. '40%'
    ax.set_ylim(rounddown(df['rand_init'].min()), 100)
    ax.set_xlim(0, 100)
    ax.invert_xaxis()
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    fig = ax.get_figure()
    # plt.show()
    json_path = os.path.join(os.getcwd(), "LTH_Results")
    fig.savefig(os.path.join(json_path, file_at))
    print("")
