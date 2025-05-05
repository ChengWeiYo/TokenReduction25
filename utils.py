import numpy as np
import pandas as pd

# SERIALS = [1, 5, 10, 11, 12, 13, 30, 31]
SERIALS_ACC = [1, 5, 10, 11, 12, 13, 14, 20, 30, 31, 60]

SERIALS_EXPLANATIONS = [
    # main experiments: 4 DS, 9 PT settings, 7 TRs, 3 KRs (and baseline with no TR/KR)
    # lr searches
    # 'lr_search_main',
    'lr_search',
    # 3 seeds
    # 'baseline_224',
    'ft_224',

    # no tr
    # 'clca_no_tr',
    'ft_224',
    # with tr (except evit, only 1 seed)
    # 'clca_tr',
    'ft_224',

    # evit (baseline and clca) for image size 224 on 4 DS
    # 'evit_baseline_clca_224',
    # 'evit_baseline_clca_224',
    'ft_224',
    'ft_224',
    # evit (baseline and clca) for image size 448 on 10 UFGIR DS
    # 'evit_baseline_clca_448',
    'ft_448',
    # evit (baseline and clca) for image size 448 on air/cub with 1 kr
    # 'evit_baseline_clca_448',
    'ft_448',

    # serial 14:
    'ft_224',
    # serial 20:
    'ft_224',

    # serial 60:
    'ft_448',

    # against intermediate features (4 DS, laion2b, no TR only)
    # 'lr_search_inter_feats',
    'lr_search',
    # 'inter_feats',
    'ft_224',

    # gradients
    'vis_gradients',

    # inference tp
    'inference_224',
    'inference_448',
]


DATASETS = [
    'aircraft',
    'cotton',
    'cub',
    'soyageing',
    'soyageingr1',
    'soyageingr3',
    'soyageingr4',
    'soyageingr5',
    'soyageingr6',
    'soygene',
    'soyglobal',
    'soylocal',
    'all'
]


METHODS = [
    'bl',

    'ifa',
    'vqt_all_lasso',
    'vqt_groups_lasso',
    'vqt_all',
    'vqt_groups',
    'h2t_all_lasso',
    'h2t_groups_lasso',
    'h2t_all',
    'h2t_groups',

    'clc',
    'cla',
    'clca_no_clr',
    'clca',
]

DATASETS_DIC = {
    'aircraft': 'Aircraft',
    'cotton': 'Cotton',
    'cub': 'CUB',
    'soyageing': 'SoyAgeing',
    'soyageingr1': 'SoyAgeingR1',
    'soyageingr3': 'SoyAgeingR3',
    'soyageingr4': 'SoyAgeingR4',
    'soyageingr5': 'SoyAgeingR5',
    'soyageingr6': 'SoyAgeingR6',
    'soygene': 'SoyGene',
    'soyglobal': 'SoyGlobal',
    'soylocal': 'SoyLocal',
    'all': 'Average',
}

TR_DIC = {
    'notr': 'No TR',
    'dyvit': 'DynamicViT',
    'evit': 'EViT',
    'topk': 'Top-K',
    'edar': 'EDAR',
    'maws': 'MAWS',
    'dmaws': 'DMAWS',
    'glsf': 'GLSF',
    'ats': 'ATS',
    'tome': 'ToMe',
    'patchmerger': 'PatchMerger',
    'sit': 'SiT',
    'dpcknn': 'DPC-KNN', 
}

PT_DIC = {
    'vit_base_patch16_224.orig_in21k': 'ViT',
    'deit_base_patch16_224.fb_in1k': 'DeiT',
    'vit_base_patch16_224_miil.in21k': 'IN21k-P',
    'vit_base_patch16_224.in1k_mocov3': 'MoCo v3',
    'vit_base_patch16_224.dino': 'DINO',
    'vit_base_patch16_224.mae': 'MAE',
    'deit3_base_patch16_224.fb_in22k_ft_in1k': 'DeiT 3 (IN21k)',
    'deit3_base_patch16_224.fb_in1k': 'DeiT 3 (IN1k)',
    'vit_base_patch16_clip_224.laion2b': 'CLIP',
    'deit_tiny_patch16_224.fb_in1k': 'Tiny',
}


METHODS_DIC = {
    'bl': 'Base',

    'ifa': 'IFA',
    'vqt_all_lasso': 'VQT (All+Lasso)',
    'vqt_groups_lasso': 'VQT (Groups+Lasso)',
    'vqt_all': 'VQT (All)',
    'vqt_groups': 'VQT (Groups)',
    'h2t_all_lasso': 'H2T (All+Lasso)',
    'h2t_groups_lasso': 'H2T (Groups+Lasso)',
    'h2t_all': 'H2T (All)',
    'h2t_groups': 'H2T (Groups)',

    'clc': 'CLC',
    'cla': 'CLA',
    'clca_no_clr': 'CLCA (No CLR)',
    'clca': 'CLCA',
}


SETTINGS_DIC = {
    'ft_224': 'IS=224',
    'ft_448': 'IS=448',
}


VAR_DIC = {
    'setting': 'Setting',

    'dataset_name': 'Dataset',
    'model': 'Model',
    'pt': 'Pretraining',
    'tr': 'Reduction',
    'kr': 'Keep Rate (%)',
    'method': 'Method',
    'family': 'Method Family',

    'acc': 'Accuracy (%)',
    'acc_std': 'Accuracy Std. Dev. (%)',

    'flops': 'FLOPs (10^9)',
    'time_train': 'Train Time (hours)',
    'vram_train': 'Train VRAM (GB)',
    'trainable_percent': 'Trainable Parameters (%)',
    'no_params': 'Number of Parameters (10^6)',
    'no_params_trainable': 'Trainable Parameters (10^6)',
    'no_params_total': 'Total Parameters (10^6)',
    'no_params_trainable_total': 'Total Trainable Params. (10^6)',
    'tp_stream': 'Stream Throughput (Images/s)',
    'vram_stream': 'Stream VRAM (GB)',
    'latency_stream': 'Stream Latency (s)',
    'tp_batched': 'Batched Throughput  (Images/s)',
    'vram_batched': 'Batched VRAM (GB)',
}


def rename_var(x):
    if x in SETTINGS_DIC.keys():
        return SETTINGS_DIC[x]

    if x in DATASETS_DIC.keys():
        return DATASETS_DIC[x]

    elif x in TR_DIC.keys():
        return TR_DIC[x]
    elif x in PT_DIC.keys():
        return PT_DIC[x]
    elif x in METHODS_DIC.keys():
        return METHODS_DIC[x]

    elif x in VAR_DIC.keys():
        return VAR_DIC[x]

    return x


def rename_vars(df, var_rename=False, args=None):
    if 'setting' in df.columns:
       df['setting'] = df['setting'].apply(rename_var)

    if 'dataset_name' in df.columns:
        df['dataset_name'] = df['dataset_name'].apply(rename_var)

    if 'tr' in df.columns:
        df['tr'] = df['tr'].apply(rename_var)
    if 'pt' in df.columns:
        df['pt'] = df['pt'].apply(rename_var)
    if 'method' in df.columns:
        df['method'] = df['method'].apply(rename_var)

    if 'family' in df.columns:
        df['family'] = df['family'].apply(rename_var)

    if var_rename:
        df.rename(columns=VAR_DIC, inplace=True)
        for k, v in VAR_DIC.items():
            if k == args.x_var_name:
                args.x_var_name = v
            elif k == args.y_var_name:
                args.y_var_name = v
            elif k == args.hue_var_name:
                args.hue_var_name = v
            elif k == args.style_var_name:
                args.style_var_name = v

    return df


def adjust_tome_kr(df):
    # tome equivalent keep rate
    df.loc[df['reduction_loc'] == '[2, 3, 5, 6, 8, 9]', 'keep_rate_single'] = 0.25
    return df


def determine_tr_pt(df):
    # token reduction methods
    df.loc[df['keep_rate_single'].isnull(), 'model'] = df['model'].apply(lambda x: f'notr_{x}')
    df.loc[df['keep_rate_single'].isnull(), 'keep_rate_single'] = 1

    # split model name into token reduction method and pretraining strategy
    df[['tr', 'pt']] = df['model'].str.split('_', n=1, expand=True)
    return df


def determine_method(df):
    conditions = [
    ((df['ifa_head'] == True) & (df['ifa_dws_conv_groups'] == 0) & (df['reduction_loc'] == '[]')),

    ((df['model'].str.contains('vqt')) & (df['reduction_loc'] == '[]') & (df['lasso_loss_weight'] == 0.0001)),
    ((df['model'].str.contains('vqt')) & (df['reduction_loc'] == '[3, 6, 9, 11]') & (df['lasso_loss_weight'] == 0.0001)),
    ((df['model'].str.contains('vqt')) & (df['reduction_loc'] == '[]') & (df['lasso_loss_weight'] == 0.0)),
    ((df['model'].str.contains('vqt')) & (df['reduction_loc'] == '[3, 6, 9, 11]') & (df['lasso_loss_weight'] == 0.0)),

    ((df['model'].str.contains('h2t')) & (df['reduction_loc'] == '[]') & (df['lasso_loss_weight'] == 0.0001)),
    ((df['model'].str.contains('h2t')) & (df['reduction_loc'] == '[3, 6, 9, 11]') & (df['lasso_loss_weight'] == 0.0001)),
    ((df['model'].str.contains('h2t')) & (df['reduction_loc'] == '[]') & (df['lasso_loss_weight'] == 0.0)),
    ((df['model'].str.contains('h2t')) & (df['reduction_loc'] == '[3, 6, 9, 11]') & (df['lasso_loss_weight'] == 0.0)),

    ((df['ifa_head'] == True) & (df['num_clr'] == 1)),
    ((df['ifa_head'] == True) & (df['clc'] == True) & (df['num_clr'] == 0)),
    ((df['ifa_head'] == True) & (df['ifa_dws_conv_groups'] != 0) & (df['clc'] == False)),
    ((df['ifa_head'] == False) & (df['clc'] == True)),

    # (df['ifa_head'] == False),
    ]
    choices = ['ifa', 'vqt_all_lasso', 'vqt_groups_lasso', 'vqt_all', 'vqt_groups',
               'h2t_all_lasso', 'h2t_groups_lasso', 'h2t_all', 'h2t_groups',
               'clca', 'clca_no_clr', 'cla', 'clc']
    df['method'] = np.select(conditions, choices, default='bl')
    return df


def add_setting(df):
    conditions = [
        (df['serial'] == 0),
        (df['serial'] == 1),

        (df['serial'] == 11),
        (df['serial'] == 12),

        (df['serial'] == 10),
        (df['serial'] == 13),
        (df['serial'] == 30),
        (df['serial'] == 31),

        (df['serial'] == 14),
        (df['serial'] == 20),

        (df['serial'] == 60),

        (df['serial'] == 3),
        (df['serial'] == 5),

        (df['serial'] == 2),

        (df['serial'] == 40),
        (df['serial'] == 41),
    ]

    df['setting'] = np.select(conditions, SERIALS_EXPLANATIONS, default='')
    return df


def standarize_df(df):
    df = adjust_tome_kr(df)

    df = determine_tr_pt(df)

    df = determine_method(df)

    df = add_setting(df)

    # rename columns for acc for simplicity
    df = df.rename(columns={'test_acc': 'acc', 'keep_rate_single': 'kr'})

    # unique combined method
    df['method_combined'] = df['method'] + '_' + df['tr'] + '_' + df['pt'] + '_' + df['kr'].astype(str)
    # df['method_combined'] = df['method'] + '_' + df['tr'] + '_' + df['pt']

    return df


def drop_columns(df, keep_acc_only=False):
    columns_to_drop = [
        'lr', 'seed', 'model', 'keep_rate', 'reduction_loc', 'ifa_head', 'clc',
        'num_clr', 'ifa_dws_conv_groups', 'lasso_loss_weight', 'input_size'
    ]
    if keep_acc_only:
        extra_cols = [
            'host', 'epochs', 'batch_size', 'num_images_train', 'num_images_val',
            'val_test_loss', 'train_acc1', 'train_loss', 
            'time_total', 'max_memory', 'no_params', 'throughput'
        ]
        columns_to_drop = columns_to_drop + extra_cols

    df = df.drop(columns=columns_to_drop)
    return df


def round_combine_str_mean_std(df, col='acc', decimals=2):
    df[f'{col}'] = df[f'{col}'].round(decimals)
    df[f'{col}_std'] = df[f'{col}_std'].round(decimals)

    df[f'{col}_mean_std_latex'] = df[f'{col}'].astype(str) + "\pm{" + df[f'{col}_std'].astype(str) + "}"
    df[f'{col}_mean_std'] = df[f'{col}'].astype(str) + "+-" + df[f'{col}_std'].astype(str)

    return df


def add_all_cols_group(df, col='dataset_name'):
    subset = df.copy(deep=False)
    subset[col] = 'all'
    df = pd.concat([df, subset], axis=0, ignore_index=True)
    return df


def filter_df(df, keep_datasets=None, keep_methods=None, keep_serials=None,
              keep_pts=None, keep_trs=None, keep_krs=None,
              filter_datasets=None, filter_methods=None, filter_serials=None,
              filter_pts=None, filter_trs=None):
    if keep_datasets:
        df = df[df['dataset_name'].isin(keep_datasets)]

    if keep_methods:
        df = df[df['method'].isin(keep_methods)]

    if keep_serials:
        df = df[df['serial'].isin(keep_serials)]

    if keep_pts:
        df = df[df['pt'].isin(keep_pts)]

    if keep_trs:
        df = df[df['tr'].isin(keep_trs)]

    if keep_krs:
        df = df[df['kr'].isin(keep_krs)]

    if filter_datasets:
        df = df[~df['dataset_name'].isin(filter_datasets)]

    if filter_methods:
        df = df[~df['method'].isin(filter_methods)]

    if filter_serials:
        df = df[~df['serial'].isin(filter_serials)]

    if filter_pts:
        df = df[~df['pt'].isin(filter_pts)]

    if filter_trs:
        df = df[~df['tr'].isin(filter_trs)]

    return df


def sort_dataset(df, method_only=False, dataset_only=False, ignore_serial=False):
    if dataset_only:
        df['dataset_order'] = pd.Categorical(df['dataset_name'], categories=DATASETS, ordered=True)

        df = df.sort_values(by=['dataset_order'], ascending=True)
        df = df.drop(columns=['dataset_order'])
    elif method_only:
        df['method_order'] = pd.Categorical(df['method'], categories=METHODS, ordered=True)

        df = df.sort_values(by=['serial', 'method_order'], ascending=True)
        df = df.drop(columns=['method_order'])
    else:
        df['dataset_order'] = pd.Categorical(df['dataset_name'], categories=DATASETS, ordered=True)
        df['method_order'] = pd.Categorical(df['method'], categories=METHODS, ordered=True)
        df['tr_order'] = pd.Categorical(df['tr'], categories=TR_DIC.keys(), ordered=True)
        df['pt_order'] = pd.Categorical(df['pt'], categories=PT_DIC.keys(), ordered=True)

        if ignore_serial:
            df = df.sort_values(by=['dataset_order', 'method_order', 'tr_order', 'pt_order', 'kr'], ascending=True)
        else:
            df = df.sort_values(by=['serial', 'dataset_order', 'method_order', 'tr_order', 'pt_order', 'kr'], ascending=True)
        df = df.drop(columns=['method_order', 'dataset_order', 'tr_order', 'pt_order'])
    return df


def fill_flops_nan(df):
    conditions = [
        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'notr'),

        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'dyvit') & (df['kr'] == 0.25),
        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'dyvit') & (df['kr'] == 0.5),
        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'dyvit') & (df['kr'] == 0.7),

        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'ats') & (df['kr'] == 0.25),
        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'ats') & (df['kr'] == 0.5),
        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'ats') & (df['kr'] == 0.7),

        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'tome') & (df['kr'] == 0.25),
        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'tome') & (df['kr'] == 0.5),
        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'tome') & (df['kr'] == 0.7),

        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'dpcknn') & (df['kr'] == 0.25),
        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'dpcknn') & (df['kr'] == 0.5),
        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'dpcknn') & (df['kr'] == 0.7),

        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'sit') & (df['kr'] == 0.25),
        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'sit') & (df['kr'] == 0.5),
        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'sit') & (df['kr'] == 0.7),

        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'patchmerger') & (df['kr'] == 0.25),
        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'patchmerger') & (df['kr'] == 0.5),
        (df['serial'] == 1) & (df['flops'].isna()) & (df['tr'] == 'patchmerger') & (df['kr'] == 0.7),
    ]

    choices = [
        max(df[(df['serial'] == 5) & (df['dataset_name'] == 'soygene') & (df['method'] == 'bl')]['flops']),

        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'dyvit') & (df['kr'] == 0.25)]['flops']),
        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'dyvit') & (df['kr'] == 0.5)]['flops']),
        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'dyvit') & (df['kr'] == 0.7)]['flops']),

        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'ats') & (df['kr'] == 0.25)]['flops']),
        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'ats') & (df['kr'] == 0.5)]['flops']),
        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'ats') & (df['kr'] == 0.7)]['flops']),

        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'tome') & (df['kr'] == 0.25)]['flops']),
        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'tome') & (df['kr'] == 0.5)]['flops']),
        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'tome') & (df['kr'] == 0.7)]['flops']),

        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'dpcknn') & (df['kr'] == 0.25)]['flops']),
        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'dpcknn') & (df['kr'] == 0.5)]['flops']),
        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'dpcknn') & (df['kr'] == 0.7)]['flops']),

        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'sit') & (df['kr'] == 0.25)]['flops']),
        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'sit') & (df['kr'] == 0.5)]['flops']),
        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'sit') & (df['kr'] == 0.7)]['flops']),

        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'patchmerger') & (df['kr'] == 0.25)]['flops']),
        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'patchmerger') & (df['kr'] == 0.5)]['flops']),
        max(df[(df['serial'] == 12) & (df['dataset_name'] == 'soygene') &
           (df['method'] == 'cla') & (df['tr'] == 'patchmerger') & (df['kr'] == 0.7)]['flops']),
    ]

    df['flops'] = np.select(conditions, choices, default=df['flops'])
    return df


'''
# compute the diffs between rows including clca and other rows 
# useful if only a particular tr/pt/kr otherwise too many diffs
def compute_diffs(df):
    ours_list = [m for m in df.index if 'clca' in m]
    methods = [m for m in df.index if 'clca' not in m]

    for ours in ours_list:
        for i, (index, row) in enumerate(df.iterrows()):
            if index != ours and index in methods:
                df.loc[f'diff_{ours}_{str(index)}'] = df.loc[df.index == ours].iloc[0] - row 

    return df
'''