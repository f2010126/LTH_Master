import pandas as pd
import os
from matplotlib import pyplot as plt


def get_df(df):
    base = df.Name == 'baseline_run'
    prune = df.Name.str.startswith('prun')
    random1 = df.Name.str.startswith('random')
    base_df = df[base]
    prune_df = df[prune]
    random_df = df[random1]
    prune_df = pd.concat([prune_df, base_df])

    return base_df, prune_df, random_df


file_names = ['All_4_3.csv', '4_4_es.csv', '4_5_64.csv', '4_5_1024.csv', '4_15_ssl.csv']


def generate_df(file_name):
    df = pd.read_csv(file_name)
    base_df, prune_df, random_df = get_df(df)

    ax = prune_df.plot(x='model_weight', y='test_acc', marker='o', label='imp')
    ax.axhline(y=base_df['test_acc'].iloc[0], color='r', linestyle='-', label='baseline')
    random_df.plot(x='model_weight', y='test_acc', marker='o', label='random', ax=ax)
    ax.set_xlabel('% of weights ')
    ax.set_ylabel('Test Accuracy')
    # xtick_freq = 0.33
    # ax.set_xticks(prune_df['model_weight'].tolist()[::int(1 / xtick_freq)])
    # ax.set_xticklabels([("{:.0f}%" if i >= 10 else "{:.1f}%").format(i) for i in
    #                     prune_df['model_weight'].tolist()[::int(1 / xtick_freq)]])
    ax.legend(loc='best')
    ax.set_xlim(0, 100)
    ax.invert_xaxis()

    fig = ax.get_figure()
    fig.savefig(os.path.join('', file_name.split('.')[0] + '.jpg'))
    plt.show()


def plot_batch():

    base64, prune64, _ = get_df(pd.read_csv('4_5_64.csv'))
    base1k, prune1k, _ = get_df(pd.read_csv('4_5_1024.csv'))

    ax = prune64.plot(x='model_weight', y='test_acc', marker='o', label='imp_64')
    prune1k.plot(x='model_weight', y='test_acc', marker='o', label='imp_1024', ax=ax)
    ax.set_xlabel('% of weights ')
    ax.set_ylabel('Test Accuracy')
    ax.legend(loc='best')
    ax.set_xlim(0, 100)
    ax.invert_xaxis()

    fig = ax.get_figure()
    fig.savefig(os.path.join('', 'Exp4_4_Batch.jpg'))
    plt.show()




if __name__ == '__main__':
    plot_batch()
    # for name in file_names:
    #     generate_df(name)
