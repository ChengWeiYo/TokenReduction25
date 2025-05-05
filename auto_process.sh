# download data from wandb
# this covers all serials except serial 2 which is only for plotting gradients
# put the individual results from 10-11-13, 30-31, and 1-5-12 into one single file: stage2_acc_train_cost
#python download_save_wandb_data.py --serials 1 5 10 11 12 13 30 31 --output_file wandb_tr_stage2_acc_train_cost.csv
python download_save_wandb_data.py --serials 10 11 13 --output_file wandb_tr_stage2_acc_train_cost_10-11-13.csv
python download_save_wandb_data.py --serials 30 31 --output_file wandb_tr_stage2_acc_train_cost_30-31.csv
python download_save_wandb_data.py --serials 1 5 12 --output_file wandb_tr_stage2_acc_train_cost_1-5-12.csv

# serial 14: 4 datasets, 9 backbones, 4 krs, 3 seeds for the new methods (324) with is=224: edar, maws, dmaws, glsf
# serial 20: 10 datasets, 9 backbones, 4 krs, 3 seeds for the new methods (1080) with is=448: edar, maws, dmaws, glsf
# serial 60 / TokenReductionTiny: 10 datasets, 1 backbone, 4 krs, 3 seeds: 120 runs for each method (7+4)
python download_save_wandb_data.py --serials 14 20 --output_file wandb_tr_stage2_acc_train_cost_14-20.csv
python download_save_wandb_data.py --serials 60 --project_name nycu_pcs/TokenReductionTiny --output_file wandb_tr_stage2_acc_train_cost_60.csv

# after downloading combine manually into wandb_tr_stage2_acc_train_cost.csv

python download_save_wandb_data.py --serials 40 41 --output_file wandb_tr_inference_cost.csv
python download_save_wandb_data.py --serials 0 3 --output_file wandb_tr_stage1_lr.csv


# wandb make scripts for stage 2 (seed) from stage 1 (lr)
python lr_script_tokenreduction.py --input_file data\wandb_tr_stage1_lr.csv --output_file script_tr_stage


# motivation (gradients instability for ufgir specially with low KRs)
# python download_plot_gradients.py --model vit_base_patch16_224.orig_in21k --method_subset clca bl --output_file vit_orig_max_grad_all_minimal
# python download_plot_gradients.py --model vit_base_patch16_224.orig_in21k --method_subset clca cla bl --output_file vit_orig_max_grad_all
# python download_plot_gradients.py --model vit_base_patch16_224.orig_in21k --method_subset clca cla clc bl --output_file vit_orig_max_grad_all_ablations

# python download_plot_gradients.py --model deit_base_patch16_224.fb_in1k --method_subset clca bl --output_file deit_max_grad_all_minimal
# python download_plot_gradients.py --model deit_base_patch16_224.fb_in1k --method_subset clca cla bl --output_file deit_max_grad_all
# python download_plot_gradients.py --model deit_base_patch16_224.fb_in1k --method_subset clca cla clc bl --output_file deit_max_grad_all_ablations

# python download_plot_gradients.py --model vit_base_patch16_224_miil.in21k --method_subset clca bl --output_file miil_max_grad_all_minimal
# python download_plot_gradients.py --model vit_base_patch16_224_miil.in21k --method_subset clca cla bl --output_file miil_max_grad_all
# python download_plot_gradients.py --model vit_base_patch16_224_miil.in21k --method_subset clca cla clc bl --output_file miil_max_grad_all_ablations

# python download_plot_gradients.py --model vit_base_patch16_224.in1k_mocov3 --method_subset clca bl --output_file moco_max_grad_all_minimal
# python download_plot_gradients.py --model vit_base_patch16_224.in1k_mocov3 --method_subset clca cla bl --output_file moco_max_grad_all
# python download_plot_gradients.py --model vit_base_patch16_224.in1k_mocov3 --method_subset clca cla clc bl --output_file moco_max_grad_all_ablations

# python download_plot_gradients.py --model vit_base_patch16_224.dino --method_subset clca bl --output_file dino_max_grad_all_minimal
# python download_plot_gradients.py --model vit_base_patch16_224.dino --method_subset clca cla bl --output_file dino_max_grad_all
# python download_plot_gradients.py --model vit_base_patch16_224.dino --method_subset clca cla clc bl --output_file dino_max_grad_all_ablations

# python download_plot_gradients.py --model vit_base_patch16_224.mae --method_subset clca bl --output_file mae_max_grad_all_minimal
# python download_plot_gradients.py --model vit_base_patch16_224.mae --method_subset clca cla bl --output_file mae_max_grad_all
# python download_plot_gradients.py --model vit_base_patch16_224.mae --method_subset clca cla clc bl --output_file mae_max_grad_all_ablations

python download_plot_gradients.py --model deit3_base_patch16_224.fb_in1k --method_subset clca bl --output_file deit3in1k_max_grad_all_minimal
python download_plot_gradients.py --model deit3_base_patch16_224.fb_in1k --method_subset clca cla bl --output_file deit3in1k_max_grad_all
python download_plot_gradients.py --model deit3_base_patch16_224.fb_in1k --method_subset clca cla clc bl --output_file deit3in1k_max_grad_all_ablations
python download_plot_gradients.py --model deit3_base_patch16_224.fb_in1k --method_subset clca cla clc bl --output_file deit3in1k_max_grad_all_ablations_colors --palette tab:blue tab:orange tab:green tab:purple --line_width 0.5

# python download_plot_gradients.py --model deit3_base_patch16_224.fb_in22k_ft_in1k --method_subset clca bl --output_file deit3in21k_max_grad_all_minimal
# python download_plot_gradients.py --model deit3_base_patch16_224.fb_in22k_ft_in1k --method_subset clca cla bl --output_file deit3in21k_max_grad_all
# python download_plot_gradients.py --model deit3_base_patch16_224.fb_in22k_ft_in1k --method_subset clca cla clc bl --output_file deit3in21k_max_grad_all_ablations

# python download_plot_gradients.py --model vit_base_patch16_clip_224.laion2b --method_subset clca bl --output_file clip_max_grad_all_minimal
# python download_plot_gradients.py --model vit_base_patch16_clip_224.laion2b --method_subset clca cla bl --output_file clip_max_grad_all
# python download_plot_gradients.py --model vit_base_patch16_clip_224.laion2b --method_subset clca cla clc bl --output_file clip_max_grad_all_ablations


# accuracies
python summarize_acc.py


# acc vs flops figure
python plot_acc_vs_flops.py --title "Accuracy vs FLOPs as Function of\nToken Keep Rate and Image Size on SoyGene" --dataset_name soygene --pt deit3_base_patch16_224.fb_in1k --output_file acc_vs_flops_soygene_deit3in1k_kr_minimal_leg_title --add_kr --methods bl clca
python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGene" --dataset_name soygene --pt deit3_base_patch16_224.fb_in1k --output_file acc_vs_flops_soygene_deit3in1k_kr_minimal_leg --add_kr --methods bl clca

#python plot_acc_vs_flops.py --title "Average Accuracy (Ten Datasets) vs FLOPs for\n Different Token Keep Rate and Image Sizes" --dataset_name all --serials 30 --pt vit_base_patch16_clip_224.laion2b --output_file acc_vs_flops_all_clip_kr_minimal_title --add_kr --methods bl clca
#python plot_acc_vs_flops.py --title "Average Accuracy (Ten Datasets) vs FLOPs for\n Different Token Keep Rate and Image Sizes" --dataset_name all --serials 30 --pt vit_base_patch16_clip_224.laion2b --output_file acc_vs_flops_all_clip_kr_title --add_kr

#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGene" --dataset_name soygene --pt deit3_base_patch16_224.fb_in1k --output_file acc_vs_flops_soygene_deit3in1k_kr_minimal --add_kr --methods bl clca
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyLocal" --dataset_name soylocal --pt deit3_base_patch16_224.fb_in1k --output_file acc_vs_flops_soylocal_deit3in1k_kr --add_kr
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyAgeing" --dataset_name soyageing --pt deit3_base_patch16_224.fb_in1k --output_file acc_vs_flops_soyageing_deit3in1k_kr_minimal --add_kr --methods bl clca

#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyLocal" --dataset_name soylocal --pt vit_base_patch16_224.orig_in21k --output_file acc_vs_flops_soylocal_orig
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGene" --dataset_name soygene --pt vit_base_patch16_224.orig_in21k --output_file acc_vs_flops_soygene_orig

#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyLocal" --dataset_name soylocal --pt deit_base_patch16_224.fb_in1k --output_file acc_vs_flops_soylocal_deit
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGene" --dataset_name soygene --pt deit_base_patch16_224.fb_in1k --output_file acc_vs_flops_soygene_deit

#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyLocal" --dataset_name soylocal --pt vit_base_patch16_224_miil.in21k --output_file acc_vs_flops_soylocal_miil
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGene" --dataset_name soygene --pt vit_base_patch16_224_miil.in21k --output_file acc_vs_flops_soygene_miil

#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyLocal" --dataset_name soylocal --pt vit_base_patch16_224.in1k_mocov3 --output_file acc_vs_flops_soylocal_moco
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGene" --dataset_name soygene --pt vit_base_patch16_224.in1k_mocov3 --output_file acc_vs_flops_soygene_moco

#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyLocal" --dataset_name soylocal --pt vit_base_patch16_224.dino --output_file acc_vs_flops_soylocal_dino
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGene" --dataset_name soygene --pt vit_base_patch16_224.dino --output_file acc_vs_flops_soygene_dino

#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyLocal" --dataset_name soylocal --pt vit_base_patch16_224.mae --output_file acc_vs_flops_soylocal_mae
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGene" --dataset_name soygene --pt vit_base_patch16_224.mae --output_file acc_vs_flops_soygene_mae

#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyLocal" --dataset_name soylocal --pt deit3_base_patch16_224.fb_in1k --output_file acc_vs_flops_soylocal_deit3in1k
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGene" --dataset_name soygene --pt deit3_base_patch16_224.fb_in1k --output_file acc_vs_flops_soygene_deit3in1k

#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyLocal" --dataset_name soylocal --pt deit3_base_patch16_224.fb_in22k_ft_in1k --output_file acc_vs_flops_soylocal_deit3in21k
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGene" --dataset_name soygene --pt deit3_base_patch16_224.fb_in22k_ft_in1k --output_file acc_vs_flops_soygene_deit3in21k

#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyLocal" --dataset_name soylocal --pt vit_base_patch16_clip_224.laion2b --output_file acc_vs_flops_soylocal_clip
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGene" --dataset_name soygene --pt vit_base_patch16_clip_224.laion2b --output_file acc_vs_flops_soygene_clip

#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on Cotton" --dataset_name cotton --serials 30 --pt vit_base_patch16_224.orig_in21k --output_file acc_vs_flops_cotton_orig
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on Cotton" --dataset_name cotton --serials 30 --pt deit_base_patch16_224.fb_in1k --output_file acc_vs_flops_cotton_deit
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on Cotton" --dataset_name cotton --serials 30 --pt vit_base_patch16_224_miil.in21k --output_file acc_vs_flops_cotton_miil
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on Cotton" --dataset_name cotton --serials 30 --pt vit_base_patch16_224.in1k_mocov3 --output_file acc_vs_flops_cotton_moco
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on Cotton" --dataset_name cotton --serials 30 --pt vit_base_patch16_224.dino --output_file acc_vs_flops_cotton_dino
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on Cotton" --dataset_name cotton --serials 30 --pt vit_base_patch16_224.mae --output_file acc_vs_flops_cotton_mae
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on Cotton" --dataset_name cotton --serials 30 --pt deit3_base_patch16_224.fb_in1k --output_file acc_vs_flops_cotton_deit3in1k
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on Cotton" --dataset_name cotton --serials 30 --pt deit3_base_patch16_224.fb_in22k_ft_in1k --output_file acc_vs_flops_cotton_deit3in21k
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on Cotton" --dataset_name cotton --serials 30 --pt vit_base_patch16_clip_224.laion2b --output_file acc_vs_flops_cotton_clip

#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGlobal" --dataset_name soyglobal --serials 30 --pt vit_base_patch16_224.orig_in21k --output_file acc_vs_flops_soyglobal_orig
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGlobal" --dataset_name soyglobal --serials 30 --pt deit_base_patch16_224.fb_in1k --output_file acc_vs_flops_soyglobal_deit
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGlobal" --dataset_name soyglobal --serials 30 --pt vit_base_patch16_224_miil.in21k --output_file acc_vs_flops_soyglobal_miil
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGlobal" --dataset_name soyglobal --serials 30 --pt vit_base_patch16_224.in1k_mocov3 --output_file acc_vs_flops_soyglobal_moco
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGlobal" --dataset_name soyglobal --serials 30 --pt vit_base_patch16_224.dino --output_file acc_vs_flops_soyglobal_dino
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGlobal" --dataset_name soyglobal --serials 30 --pt vit_base_patch16_224.mae --output_file acc_vs_flops_soyglobal_mae
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGlobal" --dataset_name soyglobal --serials 30 --pt deit3_base_patch16_224.fb_in1k --output_file acc_vs_flops_soyglobal_deit3in1k
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGlobal" --dataset_name soyglobal --serials 30 --pt deit3_base_patch16_224.fb_in22k_ft_in1k --output_file acc_vs_flops_soyglobal_deit3in21k
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyGlobal" --dataset_name soyglobal --serials 30 --pt vit_base_patch16_clip_224.laion2b --output_file acc_vs_flops_soyglobal_clip

#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyAgeing" --dataset_name soyageing --serials 30 --pt vit_base_patch16_224.orig_in21k --output_file acc_vs_flops_soyageing_orig
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyAgeing" --dataset_name soyageing --serials 30 --pt deit_base_patch16_224.fb_in1k --output_file acc_vs_flops_soyageing_deit
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyAgeing" --dataset_name soyageing --serials 30 --pt vit_base_patch16_224_miil.in21k --output_file acc_vs_flops_soyageing_miil
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyAgeing" --dataset_name soyageing --serials 30 --pt vit_base_patch16_224.in1k_mocov3 --output_file acc_vs_flops_soyageing_moco
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyAgeing" --dataset_name soyageing --serials 30 --pt vit_base_patch16_224.dino --output_file acc_vs_flops_soyageing_dino
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyAgeing" --dataset_name soyageing --serials 30 --pt vit_base_patch16_224.mae --output_file acc_vs_flops_soyageing_mae
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyAgeing" --dataset_name soyageing --serials 30 --pt deit3_base_patch16_224.fb_in1k --output_file acc_vs_flops_soyageing_deit3in1k
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyAgeing" --dataset_name soyageing --serials 30 --pt deit3_base_patch16_224.fb_in22k_ft_in1k --output_file acc_vs_flops_soyageing_deit3in21k
#python plot_acc_vs_flops.py --title "Accuracy vs FLOPs on SoyAgeing" --dataset_name soyageing --serials 30 --pt vit_base_patch16_clip_224.laion2b --output_file acc_vs_flops_soyageing_clip


# accuracy plots
# image size 224 distribution over token reduction (for different keep rates, 2 datasets: soygene and soylocal, and pt strategies)
python plot.py --keep_datasets aircraft cub soylocal soygene --keep_serials 1 10 11 12 14 --keep_methods bl clca --type_plot box --x_var_name tr --y_var_name acc --hue_var_name method --output_file box_acc_vs_tr_method_all_title --fig_size 7 4  --x_rotation 30 --title "Acc. Distribution (9 Backbones, 4 Datasets) for SotA TR Schemes" --x_label  "Token Reduction (TR) Scheme"
python plot.py --keep_datasets aircraft cub --keep_serials 1 10 11 12 14 --keep_methods bl clca --type_plot box --x_var_name tr --y_var_name acc --hue_var_name method --output_file box_acc_vs_tr_method_air_cub_title --fig_size 7 4  --x_rotation 30 --title "Acc. Distribution (9 Backbones, Aircraft & CUB) for SotA TR Schemes" --x_label  "Token Reduction (TR) Scheme"
python plot.py --keep_datasets soylocal soygene --keep_serials 1 10 11 12 14 --keep_methods bl clca --type_plot box --x_var_name tr --y_var_name acc --hue_var_name method --output_file box_acc_vs_tr_method_soy_title --fig_size 7 4  --x_rotation 30 --title "Acc. Distribution (9 Backbones, SoyLoc & SoyGene) for SotA TR Schemes" --x_label  "Token Reduction (TR) Scheme"
python plot.py --keep_datasets soylocal soygene --keep_serials 1 10 11 12 14 --keep_methods bl cla clca --type_plot box --x_var_name tr --y_var_name acc --hue_var_name method --output_file box_acc_vs_tr_method_soy_detail_title --fig_size 7 4  --x_rotation 30 --title "Acc. Distribution (9 Backbones, SoyLoc & SoyGene) for SotA TR Schemes" --x_label  "Token Reduction (TR) Scheme"

# image size 224 comparison of tr methods across datasets for different keep rates
python plot.py --keep_datasets aircraft cub soylocal soygene --keep_serials 1 10 11 12 14 --keep_methods clca --type_plot box --x_var_name dataset_name --y_var_name acc --hue_var_name tr --output_file box_acc_vs_ds_tr_clca --fig_size 7 4 --title "Acc. Dist. (9 Backbones, 3 KRs) Grouped by DS and and TR Schemes" --x_label  "Dataset Name (DS)" --loc_legend "lower left"
python plot.py --keep_datasets aircraft cub soylocal soygene --keep_serials 1 10 11 12 14 --keep_krs 0.25 --keep_methods clca --type_plot box --x_var_name dataset_name --y_var_name acc --hue_var_name tr --output_file box_acc_vs_ds_tr_clca_025 --fig_size 7 4 --title "Acc. Dist. (9 Backbones, KR=0.25) Grouped by DS and and TR Schemes" --x_label  "Dataset Name (DS)" --loc_legend "lower left"
python plot.py --keep_datasets aircraft cub soylocal soygene --keep_serials 1 10 11 12 14 --keep_krs 0.5 --keep_methods clca --type_plot box --x_var_name dataset_name --y_var_name acc --hue_var_name tr --output_file box_acc_vs_ds_tr_clca_050 --fig_size 7 4 --title "Acc. Dist. (9 Backbones, KR=0.5) Grouped by DS and and TR Schemes" --x_label  "Dataset Name (DS)" --loc_legend "lower left"
python plot.py --keep_datasets aircraft cub soylocal soygene --keep_serials 1 10 11 12 14 --keep_krs 0.7 --keep_methods clca --type_plot box --x_var_name dataset_name --y_var_name acc --hue_var_name tr --output_file box_acc_vs_ds_tr_clca_070 --fig_size 7 4 --title "Acc. Dist. (9 Backbones, KR=0.7) Grouped by DS and and TR Schemes" --x_label  "Dataset Name (DS)" --loc_legend "lower left"

python plot.py --keep_datasets aircraft cub soylocal soygene --keep_serials 1 10 11 12 14 --keep_methods bl --type_plot box --x_var_name dataset_name --y_var_name acc --hue_var_name tr --output_file box_acc_vs_ds_tr_bl --fig_size 7 4 --title "Acc. Distribution (9 Backbones, 3 KRs) Grouped by DS and and TR Schemes" --x_label  "Dataset Name (DS)" --loc_legend "lower left"
python plot.py --keep_datasets aircraft cub soylocal soygene --keep_serials 1 10 11 12 14 --keep_krs 0.25 --keep_methods bl --type_plot box --x_var_name dataset_name --y_var_name acc --hue_var_name tr --output_file box_acc_vs_ds_tr_bl_025 --fig_size 7 4 --title "Acc. Distribution (9 Backbones, KR=0.25) Grouped by DS and and TR Schemes" --x_label  "Dataset Name (DS)" --loc_legend "lower left"
python plot.py --keep_datasets aircraft cub soylocal soygene --keep_serials 1 10 11 12 14 --keep_krs 0.5 --keep_methods bl --type_plot box --x_var_name dataset_name --y_var_name acc --hue_var_name tr --output_file box_acc_vs_ds_tr_bl_050 --fig_size 7 4 --title "Acc. Distribution (9 Backbones, KR=0.5) Grouped by DS and and TR Schemes" --x_label  "Dataset Name (DS)" --loc_legend "lower left"
python plot.py --keep_datasets aircraft cub soylocal soygene --keep_serials 1 10 11 12 14 --keep_krs 0.7 --keep_methods bl --type_plot box --x_var_name dataset_name --y_var_name acc --hue_var_name tr --output_file box_acc_vs_ds_tr_bl_070 --fig_size 7 4 --title "Acc. Distribution (9 Backbones, KR=0.7) Grouped by DS and and TR Schemes" --x_label  "Dataset Name (DS)" --loc_legend "lower left"

# image size 448
python plot.py --keep_serials 30 --keep_methods bl clca --type_plot box --x_var_name kr --y_var_name acc --hue_var_name method --output_file box_acc_vs_kr_method_30_title --fig_size 6 4 --title "Acc. Distribution (9 Backbones, 10 Datasets) for KRs w/ and w/o CLCA" --x_label "Keep Rate (KR, %)"
python plot.py --keep_serials 20 30 60 --keep_methods bl cla clca --type_plot box --x_var_name kr --y_var_name acc --hue_var_name method --output_file box_acc_vs_kr_method_30_detail_title --fig_size 6 4 --title "Acc. Distribution (9 Backbones, 10 Datasets) for KRs w/ and w/o CLCA" --x_label "Keep Rate (KR, %)"

python plot.py --keep_serials 20 30 60 --keep_methods bl clca --type_plot box --x_var_name pt --y_var_name acc --hue_var_name method --output_file box_acc_vs_pt_method_30_title --fig_size 9 4 --x_rotation 30 --title "Acc. Distribution (5 Keep Rates, 10 Datasets) for SotA PT Backbones w/ and w/o CLCA" --x_label "Pretrained (PT) Backbone"
python plot.py --keep_serials 20 30 60 --keep_methods bl cla clca --type_plot box --x_var_name pt --y_var_name acc --hue_var_name method --output_file box_acc_vs_pt_method_30_detail_title --fig_size 9 4 --x_rotation 30 --title "Acc. Distribution (5 Keep Rates, 10 Datasets) for SotA PT Backbones w/ and w/o CLCA" --x_label "Pretrained (PT) Backbone"

python plot.py --keep_serials 20 30 60 --keep_methods bl clca --type_plot box --x_var_name dataset_name --y_var_name acc --hue_var_name method --output_file box_acc_vs_ds_method_30_title --fig_size 9 4 --x_rotation 30 --title "Acc. Distribution (9 Backbones, 5 KRs) for Datasets w/ and w/o CLCA" --x_label "Dataset Name (DS)"
python plot.py --keep_serials 20 30 60 --keep_methods bl cla clca --type_plot box --x_var_name dataset_name --y_var_name acc --hue_var_name method --output_file box_acc_vs_ds_method_30_detail_title --fig_size 9 4 --x_rotation 30 --title "Acc. Distribution (9 Backbones, 5 KRs) for Datasets w/ and w/o CLCA" --x_label "Dataset Name (DS)"

python plot.py --keep_serials 20 30 60 --keep_methods bl cla clca --type_plot violin --x_var_name pt --y_var_name acc --hue_var_name method --output_file violin_acc_vs_pt_method_30_detail_title --fig_size 9 4 --x_rotation 30 --title "Acc. Distribution (5 Keep Rates, 10 Datasets) for SotA PT Backbones w/ and w/o CLCA" --x_label "Pretrained (PT) Backbone"
python plot.py --keep_serials 20 30 60 --keep_methods bl cla clca --type_plot violin --x_var_name kr --y_var_name acc --hue_var_name method --output_file violin_acc_vs_kr_method_30_detail_title --fig_size 6 4 --title "Acc. Distribution (9 Backbones, 10 Datasets) for KRs w/ and w/o CLCA" --x_label "Keep Rate (KR, %)"
python plot.py --keep_serials 20 30 60 --keep_methods bl cla clca --type_plot violin --x_var_name dataset_name --y_var_name acc --hue_var_name method --output_file violin_acc_vs_ds_method_30_detail_title --fig_size 9 4 --x_rotation 30 --title "Acc. Distribution (9 Backbones, 5 Keep Rates) for Datasets w/ and w/o CLCA" --x_label "Dataset Name (DS)"

python plot.py --keep_serials 20 30 60 --keep_methods bl cla clca --type_plot box --x_var_name pt --y_var_name acc_std --hue_var_name method --output_file box_acc_stdev_vs_pt_method_30_detail_title --fig_size 9 4 --x_rotation 30 --title "Acc. Std. Dev. Distribution (5 Keep Rates, 10 Datasets) for SotA PT Backbones w/ and w/o CLCA" --x_label "Pretrained (PT) Backbone" --y_label "Acc. Std. Dev. (%)"
python plot.py --keep_serials 20 30 60 --keep_methods bl cla clca --type_plot box --x_var_name kr --y_var_name acc_std --hue_var_name method --output_file box_acc_stdev_vs_kr_method_30_detail_title --fig_size 6 4 --title "Acc. Std. Dev. Distribution (9 Backbones, 10 Datasets) for KRs w/ and w/o CLCA" --x_label "Keep Rate (KR, %)" --y_label "Acc. Std. Dev. (%)"
python plot.py --keep_serials 20 30 60 --keep_methods bl cla clca --type_plot box --x_var_name dataset_name --y_var_name acc_std --hue_var_name method --output_file box_acc_stdev_vs_ds_method_30_detail_title --fig_size 9 4 --x_rotation 30 --title "Acc. Std. Dev. Distribution (9 Backbones, 5 Keep Rates) for Datasets w/ and w/o CLCA" --x_label "Dataset Name (DS)" --y_label "Acc. Std. Dev. (%)"


# tables
# copy results for clca_evit_deit3_base_patch16_224.fb_in1k_0.1, 0.25, and 0.7
# also baseline with and without token reduction and also for deit_base and clip
# into acc_clca_highlights.csv
# original used results for clca_evit_deit3
python tably.py results\acc_flops_sota_cotton_soyageing_soyglobal_flops.csv


# visualization of attention
python vis_attention.py
python vis_attention.py --subfolder soygene --save_name dfsm_clip_soygene
python vis_attention.py --subfolder aircraft --save_name dfsm_clip_aircraft
python vis_attention.py --subfolder cub --save_name dfsm_clip_cub

python vis_attention.py --vis_all_masks --save_name dfsm_clip_soylocal_all
python vis_attention.py --subfolder soygene --save_name dfsm_clip_soygene_all --vis_all_masks
python vis_attention.py --subfolder aircraft --save_name dfsm_clip_aircraft_all --vis_all_masks
python vis_attention.py --subfolder cub --save_name dfsm_clip_cub_all --vis_all_masks

python vis_attention.py --subfolder aircraft --save_name dfsm_clip_aircraft_all_test --vis_all_masks --test_images
python vis_attention.py --subfolder cub --save_name dfsm_clip_cub_all_test --vis_all_masks --test_images
python vis_attention.py --subfolder soygene --save_name dfsm_clip_soygene_all_test --vis_all_masks --test_images
python vis_attention.py --subfolder soylocal --save_name dfsm_clip_soylocal_all_test --vis_all_masks --test_images


# missing: accuracy vs train and inference cost
