import os
import shutil

# 圖片所在資料夾
base_dir = 'results_all/plots/per_pt_bar'

# 定義正確 pt 名稱（你提供的9種）
pt_list = [
    "vit_base_patch16_224.orig_in21k",
    "deit_base_patch16_224.fb_in1k",
    "vit_base_patch16_224_miil.in21k",
    "vit_base_patch16_224.in1k_mocov3",
    "vit_base_patch16_224.dino",
    "vit_base_patch16_224.mae",
    "deit3_base_patch16_224.fb_in22k_ft_in1k",
    "deit3_base_patch16_224.fb_in1k",
    "vit_base_patch16_clip_224.laion2b",
    "deit_tiny_patch16_224.fb_in1k"
]

# 對每個圖檔進行分類
for filename in os.listdir(base_dir):
    if not filename.endswith(('.png', '.jpg', '.pdf')):
        continue

    for pt in pt_list:
        if pt in filename:
            pt_dir = os.path.join(base_dir, pt)
            os.makedirs(pt_dir, exist_ok=True)

            src_path = os.path.join(base_dir, filename)
            dst_path = os.path.join(pt_dir, filename)
            shutil.move(src_path, dst_path)
            print(f"✅ 已移動 {filename} → {pt}/")
            break
    else:
        print(f"⚠️ 找不到對應 pt，跳過：{filename}")
