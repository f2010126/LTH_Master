import json
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_graph(baseline,pruned, file_at="pruned.png"):
    df = pd.DataFrame.from_dict(pruned)
    ax = df.plot(x="rem_weight", y="val_score", title="MNSIT")
    ax.axhline(y=baseline, color='r', linestyle='-')
    fig = ax.get_figure()
    plt.show()
    main_path = os.getcwd()

    fig.savefig(os.path.join(os.getcwd(), file_at))
    print("")


