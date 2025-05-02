import os
import argparse

import pandas as pd

from utils import standarize_df, round_combine_str_mean_std, add_all_cols_group, \
    sort_dataset, filter_df, drop_columns, fill_flops_nan, DATASETS, SERIALS_ACC


def load_df(input_file):
    df = pd.read_csv(input_file)

    # methods
    df = standarize_df(df)
    return df


def aggregate_max(df, fp=None, serials=SERIALS_ACC):
    df = df[df['serial'].isin(serials)].copy(deep=False)

    # df_agg = df.groupby(['dataset_name'], as_index=False).agg({'acc': 'max'})
    df = df.loc[df.groupby('dataset_name')['acc'].idxmax()][['dataset_name', 'method_combined', 'acc', 'flops']]

    df = sort_dataset(df, dataset_only=True)

    if fp:
        df.to_csv(fp, header=True, index=False)
    return df


def aggregate_results_main(df, fp=None, serials=SERIALS_ACC,
                           cols=['serial', 'setting', 'dataset_name', 'method_combined', 'method', 'tr', 'pt', 'kr']):
    df = df[df['serial'].isin(serials)].copy(deep=False)

    # can add rows that aggregates results across all datasets for a method
    df = add_all_cols_group(df, 'dataset_name')
    # or all methods for a dataset
    # df = add_all_cols_group(df, 'method')

    df_std = df.groupby(cols, as_index=False).agg({'acc': 'std'})
    df = df.groupby(cols, as_index=False).agg({'acc': 'mean', 'flops': 'mean'})
    df['acc_std'] = df_std['acc']

    df = sort_dataset(df)

    df = round_combine_str_mean_std(df, col='acc')

    if fp:
        df.to_csv(fp, header=True, index=False)
    return df


def pivot_table(df, serial=1, fp=None, var='acc', save_diff=False):
    df = df[df['serial'] == serial].copy(deep=False)

    df = df.pivot(index='method_combined', columns='dataset_name')[var]

    datasets = [ds for ds in DATASETS if ds in df.columns]
    df = df[datasets]

    # sort by unique combined method but there's too many so just use def sort
    # df['method_order'] = pd.Categorical(df.index, categories=METHODS, ordered=True)
    # df = df.sort_values(by=['method_order'], ascending=True)
    # df = df.drop(columns=['method_order'])

    # compute_diffs may be 
    # if save_diff:
    #    df = compute_diffs(df)

    if fp:
        df.to_csv(fp, header=True, index=True)
    else:
        print(df)


def summarize_results(args):
    # load dataset and preprocess to include method, tr, pt and setting columns, rename test_acc to acc
    df = load_df(args.input_file)

    # drop columns
    df = drop_columns(df, keep_acc_only=True)

    # filter
    df = filter_df(df, getattr(args, 'keep_datasets', None), getattr(args, 'keep_methods', None))

    # fill nan for flops (serial 1)
    df = fill_flops_nan(df)

    # aggregate and save results
    fp = os.path.join(args.results_dir, args.output_file)

    max_results = aggregate_max(df, f'{fp}_max.csv', args.serials)

    df_main = aggregate_results_main(df, f'{fp}_main.csv', args.serials)

    for serial in args.serials:
        df_pivoted = pivot_table(df_main, serial, f'{fp}_{serial}_pivoted.csv',
                                 var='acc', save_diff=True)
        pivot_table(df_main, serial, f'{fp}_{serial}_pivoted_mean_std.csv',
                                 var='acc_mean_std')
        pivot_table(df_main, serial, f'{fp}_{serial}_pivoted_mean_std_latex.csv',
                                 var='acc_mean_std_latex')

    return df, df_main


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_file', type=str, 
                        default=os.path.join('data', 'wandb_tr_stage2_acc_train_cost.csv'),
                        help='filename for input .csv file from wandb')

    parser.add_argument('--keep_datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--serials', nargs='+', type=int, default=SERIALS_ACC)

    # output
    parser.add_argument('--output_file', type=str, default='acc',
                        help='filename for output .csv file')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'acc'),
                        help='The directory where results will be stored')

    args= parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    df = summarize_results(args)
    return df


if __name__ == '__main__':
    main()
