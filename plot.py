import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import sort_dataset, filter_df, rename_vars


def make_plot(args, df):
    # Seaborn Style Settings
    sns.set_theme(
        context=args.context, style=args.style, palette=args.palette,
        font=args.font_family, font_scale=args.font_scale, rc={
            "grid.linewidth": args.bg_line_width, # mod any of matplotlib rc system
            "figure.figsize": args.fig_size,
        })

    # 自動調整離群點大小：根據圖寬度與資料點數
    num_hue = len(df[args.hue_var_name].unique()) if args.hue_var_name else 1
    num_x = len(df[args.x_var_name].unique())
    scale_factor = max(1, (num_hue * num_x) / 20)  # 調整倍率
    auto_marker_size = max(1.5, 6 / scale_factor)  # 最小不要小於 1.5

    if args.type_plot == 'bar':
        ax = sns.barplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name, palette="tab20", data=df)
    elif args.type_plot == 'box':
        ax = sns.boxplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name, palette="tab20", data=df)
    elif args.type_plot == 'violin':
        ax = sns.violinplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name, palette="tab20", data=df)
    elif args.type_plot == 'line':
        ax = sns.lineplot(x=args.x_var_name, y=args.y_var_name, marker=args.marker,
                          hue=args.hue_var_name, palette="tab20", style=args.style_var_name,
                          markers=True, linewidth=args.line_width, data=df)
    elif args.type_plot == 'scatter':
        ax = sns.scatterplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name, palette="tab20",
                             style=args.style_var_name, size=args.size_var_name,
                             sizes=tuple(args.sizes), legend='brief', data=df)
    else:
        raise NotImplementedError
    
    # 強制調整所有離群點標記大小與顏色
    for line in ax.lines:
        if line.get_marker() == 'o':
            line.set_markersize(auto_marker_size)
            line.set_markerfacecolor('none')

    # labels and title
    ax.set(xlabel=args.x_label, ylabel=args.y_label, title=args.title, ylim=args.y_lim)

    if args.log_scale_x:
        ax.set_xscale('log')
    if args.log_scale_y:
        ax.set_yscale('log')

    # ticks labels
    if args.x_ticks_labels:
        x_ticks = ax.get_xticks() if getattr(args, 'x_ticks', None) is None else args.x_ticks
        ax.set_xticks(x_ticks , labels=args.x_ticks_labels)

    # Rotate x-axis or y-axis ticks lables
    if (args.x_rotation != None):
        plt.xticks(rotation = args.x_rotation)
    if (args.y_rotation != None):
        plt.yticks(rotation = args.y_rotation)

    # Change location of legend
    if args.hue_var_name:
        sns.move_legend(ax, loc=args.loc_legend, ncol=args.legend_ncol, fontsize=8)

    # save plot
    output_file = os.path.join(args.results_dir, f'{args.output_file}.{args.save_format}')
    plt.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
    print('Save plot to directory ', output_file)

    plt.clf()

    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # Subset models and datasets
    parser.add_argument('--input_file', type=str,
                        default=os.path.join('results_all', 'acc', 'acc_no_avg.csv'),
                        help='filename for input .csv file')

    parser.add_argument('--keep_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--keep_methods', nargs='+', type=str, default=None)
    parser.add_argument('--keep_serials', nargs='+', type=int, default=None)
    parser.add_argument('--keep_pts', nargs='+', type=str, default=None)
    parser.add_argument('--keep_trs', nargs='+', type=str, default=None)
    parser.add_argument('--keep_krs', nargs='+', type=float, default=None)
    parser.add_argument('--filter_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--filter_methods', nargs='+', type=str, default=None)
    parser.add_argument('--filter_serials', nargs='+', type=str, default=None)
    parser.add_argument('--filter_pts', nargs='+', type=str, default=None)
    parser.add_argument('--filter_trs', nargs='+', type=str, default=None)

    # Make a plot
    parser.add_argument('--log_scale_x', action='store_true')
    parser.add_argument('--log_scale_y', action='store_true')
    parser.add_argument('--type_plot', choices=['bar', 'line', 'box', 'violin', 'scatter'],
                        default='box', help='the type of plot (line, bar)')
    parser.add_argument('--x_var_name', type=str, default='method',
                        help='name of the variable for x')
    parser.add_argument('--y_var_name', type=str, default='acc',
                        help='name of the variable for y')
    parser.add_argument('--hue_var_name', type=str, default=None,
                        help='legend of this bar plot')
    parser.add_argument('--style_var_name', type=str, default=None,
                        help='legend of this bar plot')
    parser.add_argument('--size_var_name', type=str, default=None,)
    parser.add_argument('--orient', type=str, default=None,
                        help='orientation of plot "v", "h"')

    # output
    parser.add_argument('--output_file', default='acc_vs_method', type=str,
                        help='File path')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'plots'),
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
    parser.add_argument('--font_scale', type=int, default=0.8,
                        help='adjust the scale of the fonts')
    parser.add_argument('--bg_line_width', type=int, default=0.25,
                        help='adjust the scale of the line widths')
    parser.add_argument('--line_width', type=int, default=0.75,
                        help='adjust the scale of the line widths')
    parser.add_argument('--fig_size', nargs='+', type=float, default=[6, 4],
                        help='size of the plot')
    parser.add_argument('--sizes', type=int, nargs='+', default=[40, 800])
    parser.add_argument('--marker', type=str, default='o',
                        help='type of marker for line plot ".", "o", "^", "x", "*"')
    parser.add_argument('--dpi', type=int, default=300)

    # Set title, labels and ticks
    parser.add_argument('--title', type=str,
                        default='Average Accuracy over Ten UFGIR Datasets vs FLOPs',
                        help='title of the plot')
    parser.add_argument('--x_label', type=str, default='FLOPs (10^9)',
                        help='x label of the plot')
    parser.add_argument('--y_label', type=str, default='Accuracy (%)',
                        help='y label of the plot')
    parser.add_argument('--y_lim', nargs='*', type=int, default=None,
                        help='limits for y axis (suggest --ylim 0 100)')
    parser.add_argument('--x_ticks', nargs='+', type=int, default=None)
    parser.add_argument('--x_ticks_labels', nargs='+', type=str, default=None,
                        help='labels of x-axis ticks')
    parser.add_argument('--x_rotation', type=int, default=None,
                        help='lotation of x-axis lables')
    parser.add_argument('--y_rotation', type=int, default=None,
                        help='lotation of y-axis lables')

    # Change location of legend
    parser.add_argument('--loc_legend', type=str, default='upper right',
                        help='location of legend options are upper, lower, left right, center')
    parser.add_argument('--legend_ncol', type=int, default=1,
                    help='number of columns in legend')

    args= parser.parse_args()
    return args


def preprocess_df(args):
    df = pd.read_csv(args.input_file)
    print(len(df))
    df = filter_df(
        df,
        getattr(args, 'keep_datasets', None),
        getattr(args, 'keep_methods', None),
        getattr(args, 'keep_serials', None),
        getattr(args, 'keep_pts', None),
        getattr(args, 'keep_trs', None),
        getattr(args, 'keep_krs', None),
        getattr(args, 'filter_datasets', None),
        getattr(args, 'filter_methods', None),
        getattr(args, 'filter_serials', None),
        getattr(args, 'filter_pts', None),
        getattr(args, 'filter_trs', None),
    )
    print(len(df))

    df = sort_dataset(df, ignore_serial=True)

    df = rename_vars(df, var_rename=True, args=args)
    # print(df)
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