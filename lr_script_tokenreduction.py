import os
import argparse
import pandas as pd

def make_lr_script(args):
    # create file with specified file name
    os.makedirs(args.results_dir, exist_ok=True)
    output_file = os.path.join(args.results_dir, f'{args.output_file}.sh')
    f = open(output_file, "w")

    df = pd.read_csv(args.input_file)
    df = df[['dataset_name', 'model', 'keep_rate', 'reduction_loc',
                'ifa_head', 'clc', 'num_clr', 'ifa_dws_conv_groups', 'lasso_loss_weight',
                'test_acc', 'train_acc1', 'lr']]


    df = df.fillna({'ifa_head': '', 'clc': '', 'num_clr': '', 'ifa_dws_conv_groups': '', 'lasso_loss_weight': ''})

    df['ifa_head'] = df['ifa_head'].apply(lambda x: f'_cla' if x else '')
    df['clc'] = df['clc'].apply(lambda x: f'_clc' if x else '')
    df['num_clr'] = df['num_clr'].apply(lambda x: f'_{x}' if x else '')
    df['ifa_dws_conv_groups'] = df['ifa_dws_conv_groups'].apply(lambda x: f'_ifa' if x == 0 else '')
    df['lasso_loss_weight'] = df['lasso_loss_weight'].apply(lambda x: f'_lasso' if x else '')

    df['method'] = df['model'] + df['ifa_head'] + df['clc'] + df['num_clr'] + df['ifa_dws_conv_groups'] + df['lasso_loss_weight']


    #for loop for each dataset
    for dataset in df['dataset_name'].unique():

        # for loop for each method
        for method in df[df['dataset_name'] == dataset]['method'].unique():

            # token reduction loop        
            locs = df[
                (df['dataset_name'] == dataset) &
                (df['method'] == method)
            ]['reduction_loc'].unique()

            for loc in locs:
                krs = locs = df[
                    (df['dataset_name'] == dataset) &
                    (df['method'] == method) &
                    (df['reduction_loc'] == loc)
                ]['keep_rate'].unique()

                # keep rate loop
                for keep_rate in krs:

                    # filter the subset of the dataset based on the dataset and the method
                    df_subset = df[
                        (df['dataset_name'] == dataset) &
                        (df['method'] == method) &
                        (df['reduction_loc'] == loc) &
                        (df['keep_rate'] == keep_rate)
                    ].copy(deep=False)
                    df_subset.dropna(subset=args.selection_var, inplace=True)
                    
                    # sort subset based on val acc/loss (lowest to highest)
                    ascending = True if 'loss' in args.selection_var else False
                    df_subset_sorted = df_subset.sort_values(by=[args.selection_var], ascending=ascending)

                    # get index and train acc corresponding to best val acc (0)
                    lr = df_subset_sorted.iloc[0]['lr']

                    # check the current subset and the selected LR
                    # print(dataset, method, lr, df_subset.head())


                    #write to file
                    model = df_subset['model'].iloc[0]
                    fz = ' --freeze_backbone' if 'fz' in method else ''

                    if loc == '[]':
                        rl = ''
                    else:
                        rl = loc.replace('[', '').replace(']', '').replace(',', '')
                        rl = f' --reduction_loc {rl}'

                    if keep_rate == '[]':
                        kr = ''
                    else:
                        kr = keep_rate.replace('[', '').replace(']', '')
                        kr = f' --keep_rate {kr}'

                    lasso = df_subset['lasso_loss_weight'].iloc[0]
                    if lasso:
                        lasso = f' --lasso_loss_weight 1e-4'
                    else:
                        lasso = ''

                    cla = df_subset['ifa_head'].iloc[0]
                    ifa = df_subset['ifa_dws_conv_groups'].iloc[0]
                    if ifa:
                        head = ' --ifa_head --ifa_dws_conv_groups 0'
                    elif cla:
                        head = ' --ifa_head'
                    else:
                        head = ''

                    clc = df_subset['clc'].iloc[0]
                    if clc:
                        clc = ' --clc'
                    else:
                        clc = ''

                    num_clr = df_subset['num_clr'].iloc[0].replace('_', '', 1)
                    if num_clr:
                        num_clr = f' --num_clr {num_clr}'
                    else:
                        num_clr = ''

                    method_text = f'{fz}{rl}{kr}{head}{clc}{num_clr}{lasso}'

                    #write to file
                    # for seed in (1, 10, 100):
                    for seed in [1]:
                        line = f'{args.prefix} --seed {seed} --cfg configs/{dataset}_ft_weakaugs.yaml --model {model} --lr {lr}{method_text}\n'
                        f.write(line)

                f.write('\n')

            f.write('\n')

        f.write('\n')

    f.close()

    return 0


def parse_args():

    parser = argparse.ArgumentParser()

    #parser arguments
    parser.add_argument('--input_file', type=str,
                        default=os.path.join('data', 'wandb_tr_stage1_lr.csv'),
                        help='filename for input .csv file')

    parser.add_argument('--selection_var', type=str, default='test_acc')
    parser.add_argument('--prefix', type=str,
                        default='python -u train.py --serial 1 --train_trainval --num_workers 8',
                        help='prefix for the file in each line')
    parser.add_argument('--output_file', type=str, default='script_tr_stage2',
                        help='output file name')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'lr_scripts'),
                        help='The directory where results will be stored')

    args= parser.parse_args()

    print(args)

    return args


def main():
    args = parse_args()

    make_lr_script(args)

    return 0


if __name__ == '__main__':
    main()
