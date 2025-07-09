import pandas as pd

# 載入資料
df = pd.read_csv('results_all/acc/acc_no_avg.csv')
df = df[df['method'] == 'clca']
df = df[df['kr'].isin([0.1, 0.25, 0.5, 0.7, 0.9])]
# Those tr methods all have 4 keep rates
df = df[df['tr'].isin(['edar', 'nfedar', 'maws', 'dmaws', 'glsf', 'evit', 'dpcknn'])]
# Serial1: only BL
# Serial11: No TR
# Serial14: No BL
df = df[df['serial'].isin([10, 12, 13, 14])]
# df = df[df['serial'].isin([20, 30])]
# df = df[df['serial'].isin([60])]

# 加上 kr（keep rate）分組，計算 mean / std
grouped = df.groupby(['dataset_name', 'kr', 'tr', 'pt'])['acc'].mean().reset_index()
grouped_std = df.groupby(['dataset_name', 'kr', 'tr', 'pt'])['acc'].std().reset_index().rename(columns={'acc': 'acc_std'})

# 找出每個 dataset + kr + tr 下的最佳 pt
best_backbones = grouped.loc[grouped.groupby(['dataset_name', 'kr', 'tr'])['acc'].idxmax()]
best_backbones = best_backbones.merge(grouped_std, on=['dataset_name', 'kr', 'tr', 'pt'])

best_tr_per_kr = best_backbones.loc[best_backbones.groupby(['dataset_name', 'kr'])['acc'].idxmax()]

# 1. 先對每個 dataset + tr 平均這個 TR 下所有 KR 的最佳 acc（已搭配 best pt）
avg_acc_per_tr = best_backbones.groupby(['dataset_name', 'tr']).agg({
    'acc': 'mean',
    'pt': lambda x: x.value_counts().idxmax(),  # 取最多次出現的 pt 當作代表 (可改成 max acc pt)
    'acc_std': 'mean'
}).reset_index()

# 繼續使用前面算出的 avg_acc_per_tr
avg_acc_per_tr['rank'] = avg_acc_per_tr.groupby('dataset_name')['acc'].rank(ascending=False, method='min')

# 依照 dataset 和 rank 排序
avg_acc_per_tr = avg_acc_per_tr.sort_values(['dataset_name', 'rank'])

# 印出
for _, row in best_backbones.iterrows(): 
    dataset = row['dataset_name']
    kr = row['kr']
    tr = row['tr']
    pt = row['pt']
    avg_acc = row['acc']
    std = row['acc_std']
    print(f"Dataset: {dataset} | kr: {kr} | tr: {tr} | Best Backbone: {pt} | Avg Acc: {avg_acc:.2f}% ± {std:.2f}%")

# 印出
print("== Best TR (w/ best backbone) under each Dataset + KR ==")
for _, row in best_tr_per_kr.iterrows():
    dataset = row['dataset_name']
    kr = row['kr']
    tr = row['tr']
    pt = row['pt']
    avg_acc = row['acc']
    std = row['acc_std']
    print(f"Dataset: {dataset} | KR: {kr} | Best TR: {tr} | Backbone: {pt} | Acc: {avg_acc:.2f}% ± {std:.2f}%")

# 印出所有排名
print("== TR Method Rankings per Dataset (avg across KRs, w/ best backbone) ==")
for _, row in avg_acc_per_tr.iterrows():
    dataset = row['dataset_name']
    tr = row['tr']
    pt = row['pt']
    avg_acc = row['acc']
    std = row['acc_std']
    rank = int(row['rank'])
    print(f"Dataset: {dataset} | Rank: {rank} | TR: {tr} | Backbone: {pt} | Avg Acc: {avg_acc:.2f}% ± {std:.2f}%")


# 計算每個 TR 的平均排名與參與次數
avg_rank_per_tr = avg_acc_per_tr.groupby('tr').agg(
    avg_rank=('rank', 'mean'),
    count=('rank', 'count')
).reset_index().sort_values('avg_rank')

# 印出結果
print("== Average Rank and Count of TR Methods Across Datasets ==")
for _, row in avg_rank_per_tr.iterrows():
    tr = row['tr']
    avg_rank = row['avg_rank']
    count = row['count']
    print(f"TR: {tr:<8} | Avg Rank: {avg_rank:.2f} | Count: {count}")


# 比較 clca vs bl 的平均 acc 差異
df_all = pd.read_csv('results_all/acc/acc_no_avg.csv')
# df_all = df_all[df_all['tr'].isin(['edar', 'nfedar', 'maws', 'dmaws', 'glsf', 'evit', 'dpcknn'])]
# df_all = df_all[df_all['kr'].isin([0.1, 0.25, 0.5, 0.7, 0.9])]
df_all = df_all[df_all['serial'].isin([1, 10, 11, 12])]

# 平均 acc for clca
mean_acc_clca = df_all[df_all['method'] == 'clca']['acc'].mean()
# 平均 acc for bl
mean_acc_bl = df_all[df_all['method'] == 'bl']['acc'].mean()

# 差距
diff = mean_acc_clca - mean_acc_bl

print("\n== Average Acc Comparison: CLCA vs BL ==")
print(f"CLCA Avg Acc: {mean_acc_clca:.2f}%")
print(f"BL   Avg Acc: {mean_acc_bl:.2f}%")
print(f"Difference : {diff:.2f}% ↑")

# 每個 dataset 上 clca vs bl 的平均差
df_grouped = df_all[df_all['method'].isin(['clca', 'bl'])]
avg_by_dataset = df_grouped.groupby(['dataset_name', 'method'])['acc'].mean().unstack()
avg_by_dataset['diff'] = avg_by_dataset['clca'] - avg_by_dataset['bl']

print("\n== Per Dataset CLCA vs BL Avg Acc ==")
print(avg_by_dataset.round(2))