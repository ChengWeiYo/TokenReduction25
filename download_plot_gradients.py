import os
import argparse
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import standarize_df, rename_vars


def get_serial_2_runs():
    api = wandb.Api()

    runs = api.runs(path='nycu_pcs/TokenReductionPT', filters={'$and': [
        {'config.serial': 2},
    ]})

    print('Downloaded runs: ', len(runs))
    return runs


def get_gradients_df(runs):
    run = runs[0]
    history_cols = [c for c in run.history().columns if 'grad' in c] + ['_step']
    cfg_cols = ['serial', 'model', 'dataset_name', 'reduction_loc', 'keep_rate_single',
                'ifa_head', 'clc', 'num_clr', 'lasso_loss_weight', 'ifa_dws_conv_groups']

    df = []

    for run in runs:
        # just keep until the previous to last value since the last value is nan
        history_temp = run.history()[history_cols].iloc[:-1]
        cfg = {col: run.config.get(col, None) for col in cfg_cols}
        cfg['reduction_loc'] = str(cfg['reduction_loc'])
        summary = {col: run.summary.get(col, None) for col in ['test_acc']}

        history_temp = history_temp.assign(**cfg)
        history_temp = history_temp.assign(**summary)

        df.append(history_temp)

    df = pd.concat(df, ignore_index=True)

    print(df.head())
    return df


def modify_df(df, args):

    df = standarize_df(df)

    df = df[df['pt'] == args.model]
    if hasattr(args, 'method_subset') and args.method_subset:
        df = df[df['method'].isin(args.method_subset)]
        df['method_order'] = pd.Categorical(df['method'], categories=args.method_subset, ordered=True)
        df = df.sort_values(by=['method_order'], ascending=True)

    df['acc'] = df['acc'].round(decimals=1).astype(str)

    df = rename_vars(df, var_rename=False, args=args)

    # df['Method: Top-1 Accuracy (%)'] = df['tr'] + df['method'] + ': ' + df['acc']
    df[args.hue_var_name] = df['method'] + ': ' + df['acc']
    print(df.head())

    return df


def make_plot(args, df):
    # Seaborn Style Settings
    sns.set_theme(
        context=args.context, style=args.style, palette=args.palette,
        font=args.font_family, font_scale=args.font_scale, rc={
            "grid.linewidth": args.bg_line_width, # mod any of matplotlib rc system
            "figure.figsize": args.fig_size,
        })

    ax = sns.lineplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name,
                      linewidth=args.line_width, data=df)

    # Remove top, right border by default
    # sns.despine(top=False, right=True, left=False, bottom=False)

    # labels and title
    ax.set(xlabel=args.x_label, ylabel=args.y_label, title=args.title, ylim=args.y_lim)

    # ticks labels
    if args.xticks_labels:
        x_ticks = ax.get_xticks() if args.xticks_values is None else args.xticks_values
        ax.set_xticks(x_ticks, labels=args.xticks_labels)

    # Rotate x-axis or y-axis ticks lables
    if (args.x_rotation != None):
        plt.xticks(rotation = args.x_rotation)
    if (args.y_rotation != None):
        plt.yticks(rotation = args.y_rotation)

    # Change location of legend
    if args.hue_var_name:
        sns.move_legend(ax, loc=args.loc_legend)

    # save plot
    output_file = os.path.join(args.results_dir, f'{args.output_file}.{args.save_format}')
    plt.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
    print('Save plot to directory ', output_file)

    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # Subset models and datasets
    parser.add_argument('--model', type=str,
                        default='deit3_base_patch16_224.fb_in1k',
                        help='name of the variable for x')
    parser.add_argument('--method_subset', type=str, nargs='+',
                        default=['bl', 'cla', 'clca'])

    # Make a plot
    parser.add_argument('--x_var_name', type=str, default='_step',
                        help='name of the variable for x')
    parser.add_argument('--y_var_name', type=str, default='max_grad_all',
                        choices=['std_max_grad_layer', 'max_grad_all',
                                 'mean_avg_grad_layer', 'mean_max_grad_layer',
                                 'std_avg_grad_layer'],
                        help='name of the variable for y')
    parser.add_argument('--hue_var_name', type=str, default='Method: Acc. (%)',
                        help='legend of this bar plot')
    parser.add_argument('--orient', type=str, default=None,
                        help='orientation of plot "v", "h"')

    # output
    parser.add_argument('--output_file', default='deit3_in1k_max_grad_all', type=str,
                        help='File path')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'gradients'),
                        help='The directory where results will be stored')
    parser.add_argument('--save_format', choices=['pdf', 'png', 'jpg'], default='png', type=str,
                        help='Print stats on word level if use this command')

    # style related
    parser.add_argument('--context', type=str, default='notebook',
                        help='''affects font sizes and line widths
                        # notebook (def), paper (small), talk (med), poster (large)''')
    parser.add_argument('--style', type=str, default='whitegrid',
                        help='''affects plot bg color, grid and ticks
                        # whitegrid (white bg with grids), 'white', 'darkgrid', 'ticks'
                        ''')
    parser.add_argument('--palette', type=str, nargs='+', default='colorblind',
                        help='''
                        color palette (overwritten by color)
                        # None (def), 'pastel', 'Blues' (blue tones), 'colorblind'
                        # can create a palette that highlights based on a category
                        can create palette based on conditions
                        pal = {"versicolor": "g", "setosa": "b", "virginica":"m"}
                        pal = {species: "r" if species == "versicolor" else "b" for species in df.species.unique()}
                        ''')
    parser.add_argument('--color', type=str, default=None)
    parser.add_argument('--font_family', type=str, default='serif',
                        help='font family (sans-serif or serif)')
    parser.add_argument('--font_scale', type=int, default=0.8, # 0.8 originally
                        help='adjust the scale of the fonts')
    parser.add_argument('--bg_line_width', type=int, default=0.1,
                        help='adjust the scale of the line widths')
    parser.add_argument('--line_width', type=float, default=0.75, # 0.75 originally
                        help='adjust the scale of the line widths')
    parser.add_argument('--fig_size', nargs='+', type=float, default=[4, 3], # [6, 4]
                        help='size of the plot')
    parser.add_argument('--marker', type=str, default='.',
                        help='type of marker for line plot ".", "o", "^", "x", "*"')
    parser.add_argument('--dpi', type=int, default=300)

    # Set title, labels and ticks
    parser.add_argument('--title', type=str,
                        default='Max Gradient Absolute Value vs Train Step',
                        # on SoyLocal for DeiT3 EViT with KR=10%
                        help='title of the plot')
    parser.add_argument('--x_label', type=str, default='Step',
                        help='x label of the plot')
    parser.add_argument('--y_label', type=str, default='Gradient Magnitude',
                        help='y label of the plot')
    parser.add_argument('--y_lim', nargs='*', type=int, default=None,
                        help='limits for y axis (suggest --ylim 0 100)')
    parser.add_argument('--xticks_labels', nargs='+', type=str, default=None,
                        help='labels of x-axis ticks')
    parser.add_argument('--x_rotation', type=int, default=None,
                        help='lotation of x-axis lables')
    parser.add_argument('--y_rotation', type=int, default=None,
                        help='lotation of y-axis lables')

    # Change location of legend
    parser.add_argument('--loc_legend', type=str, default='upper right',
                        help='location of legend options are upper, lower, left right, center')

    args= parser.parse_args()
    return args


def main():
    args = parse_args()
    args.title = args.title.replace('\\n', '\n')
    os.makedirs(args.results_dir, exist_ok=True)

    if args.color:
        # single color for whole palette (sns defaults to 6 colors)
        args.palette = [args.color for _ in range(len(args.subset_models))]

    runs = get_serial_2_runs()

    df = get_gradients_df(runs)

    df = modify_df(df, args)

    make_plot(args, df)

    return 0

if __name__ == '__main__':
    main()