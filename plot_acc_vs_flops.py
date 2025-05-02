import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import rename_vars, METHODS


def make_plot(args, df):
    # Seaborn Style Settings
    sns.set_theme(
        context=args.context, style=args.style, palette=args.palette,
        font=args.font_family, font_scale=args.font_scale, rc={
            "grid.linewidth": args.bg_line_width, # mod any of matplotlib rc system
            "figure.figsize": args.fig_size,
        })

    # ax = sns.lineplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name,
    #                  style=args.style_var_name, markers=True, linewidth=args.line_width, data=df)
    ax = sns.lineplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name,
                      marker=args.marker, markers=True, linewidth=args.line_width, data=df)

    # Add text at each data point
    if args.add_kr:
        for i, row in df.iterrows():
            if (row[args.hue_var_name] == "Base"):
                if row['Setting'] == 'IS=224' and (row['Keep Rate (%)'] == 0.1):
                    ax.text(row[args.x_var_name], row[args.y_var_name] - 2, f"{int(row['Keep Rate (%)'] * 100)} ({row['Setting'].replace('IS=', '')})",
                            color='black', ha='center', va='bottom', fontsize=args.font_size)
                elif row['Setting'] == 'IS=224' and (row['Keep Rate (%)'] == 0.7):
                   ax.text(row[args.x_var_name], row[args.y_var_name] - 1.25, f"{int(row['Keep Rate (%)'] * 100)} ({row['Setting'].replace('IS=', '')})",
                           color='black', ha='center', va='bottom', fontsize=args.font_size)
                elif row['Setting'] == 'IS=448' and (row['Keep Rate (%)'] == 0.1):
                   ax.text(row[args.x_var_name], row[args.y_var_name] - 2, f"{int(row['Keep Rate (%)'] * 100)} ({row['Setting'].replace('IS=', '')})",
                           color='black', ha='center', va='bottom', fontsize=args.font_size)
                # elif row['Setting'] == 'IS=224' and (row['Keep Rate (%)'] == 0.5):
                #    ax.text(row[args.x_var_name] - 0.5, row[args.y_var_name] - 0.5, f"{int(row['Keep Rate (%)'] * 100)} ({row['Setting'].replace('IS=', '')})",
                #            color='black', ha='center', va='bottom', fontsize=args.font_size)
                # elif row['Setting'] == 'IS=224':
                #    ax.text(row[args.x_var_name], row[args.y_var_name], f"{int(row['Keep Rate (%)'] * 100)} ({row['Setting'].replace('IS=', '')})",
                #            color='black', ha='center', va='bottom', fontsize=args.font_size)
                else:
                    ax.text(row[args.x_var_name], row[args.y_var_name], f"{int(row['Keep Rate (%)'] * 100)} ({row['Setting'].replace('IS=', '')})",
                            color='black', ha='center', va='bottom', fontsize=args.font_size)

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
    parser.add_argument('--input_file', type=str,
                        default=os.path.join('results_all', 'acc', 'acc_main.csv'),
                        help='filename for input .csv file')
    parser.add_argument('--dataset_name', type=str, default='soylocal')
    parser.add_argument('--pt', type=str, default='deit3_base_patch16_224.fb_in1k')
    parser.add_argument('--serials', nargs='+', type=int,
                        default=[1, 10, 11, 13, 30, 31])
    parser.add_argument('--methods', nargs='+', type=str, default=['bl', 'cla', 'clca'])
    parser.add_argument('--tr', nargs='+', type=int, default=['notr', 'evit'])

    parser.add_argument('--add_kr', action='store_true')

    # Make a plot
    parser.add_argument('--x_var_name', type=str, default='flops',
                        help='name of the variable for x')
    parser.add_argument('--y_var_name', type=str, default='acc',
                        help='name of the variable for y')
    parser.add_argument('--hue_var_name', type=str, default='method',
                        help='legend of this bar plot')
    parser.add_argument('--style_var_name', type=str, default=None,
                        help='legend of this bar plot')
    parser.add_argument('--orient', type=str, default=None,
                        help='orientation of plot "v", "h"')

    # output
    parser.add_argument('--output_file', default='acc_vs_flops_soylocal_deit3in1k', type=str,
                        help='File path')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'acc_vs_cost'),
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
    parser.add_argument('--palette', type=str, default='colorblind',
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
    parser.add_argument('--font_size', type=int, default=7)
    parser.add_argument('--font_scale', type=int, default=0.8,
                        help='adjust the scale of the fonts')
    parser.add_argument('--bg_line_width', type=int, default=0.25,
                        help='adjust the scale of the line widths')
    parser.add_argument('--line_width', type=int, default=0.75,
                        help='adjust the scale of the line widths')
    parser.add_argument('--fig_size', nargs='+', type=float, default=[4, 3],
                        help='size of the plot')
    parser.add_argument('--marker', type=str, default='o',
                        help='type of marker for line plot ".", "o", "^", "x", "*"')
    parser.add_argument('--dpi', type=int, default=300)

    # Set title, labels and ticks
    parser.add_argument('--title', type=str,
                        default='Accuracy vs FLOPs on SoyLocal',
                        help='title of the plot')
    parser.add_argument('--x_label', type=str, default='FLOPs (10^9)',
                        help='x label of the plot')
    parser.add_argument('--y_label', type=str, default='Top-1 Accuracy (%)',
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
    parser.add_argument('--loc_legend', type=str, default='lower right',
                        help='location of legend options are upper, lower, left right, center')

    args= parser.parse_args()
    return args


def print_method_min_max(df):
    for m in df['method'].unique():
        print(m)
        print('Min: ', min(df[df['method'] == m]['acc']))
        print('Max: ', max(df[df['method'] == m]['acc']))
    return 0


def preprocess_df(args, tr=['notr', 'evit']):
    df = pd.read_csv(args.input_file)

    df = df[(df['serial'].isin(args.serials)) & (df['tr'].isin(tr)) &
            (df['method'].isin(args.methods)) &
            (df['dataset_name'] == args.dataset_name) & (df['pt'] == args.pt)].copy(deep=False)

    df['method_order'] = pd.Categorical(df['method'], categories=METHODS, ordered=True)
    df = df.sort_values(by=['method_order'], ascending=True)

    print_method_min_max(df)

    df = rename_vars(df, var_rename=True, args=args)

    return df


def main():
    args = parse_args()
    args.title = args.title.replace("\\n", "\n")
    os.makedirs(args.results_dir, exist_ok=True)

    if args.color:
        # single color for whole palette (sns defaults to 6 colors)
        args.palette = [args.color for _ in range(len(args.subset_models))]

    df = preprocess_df(args)

    make_plot(args, df)

    return 0

if __name__ == '__main__':
    main()