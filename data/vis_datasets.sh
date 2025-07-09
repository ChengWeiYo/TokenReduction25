# visualize datasets (class agnostic)
python -u tools/vis_dataset.py --debugging --cfg configs/leaves_ft_weakaugs.yaml --seed 1 --batch_size 12 --serial 420 --vis_cols 12

python -u tools/vis_dataset.py --debugging --cfg configs/cotton_ft_weakaugs.yaml --seed 1 --batch_size 12 --serial 420 --vis_cols 12
python -u tools/vis_dataset.py --debugging --cfg configs/soyageing_ft_weakaugs.yaml --seed 1 --batch_size 12 --serial 420 --vis_cols 12
python -u tools/vis_dataset.py --debugging --cfg configs/soygene_ft_weakaugs.yaml --seed 1 --batch_size 12 --serial 420 --vis_cols 12
python -u tools/vis_dataset.py --debugging --cfg configs/soyglobal_ft_weakaugs.yaml --seed 1 --batch_size 12 --serial 420 --vis_cols 12
python -u tools/vis_dataset.py --debugging --cfg configs/soylocal_ft_weakaugs.yaml --seed 1 --batch_size 12 --serial 420 --vis_cols 12

# visualize datasets per class
python -u tools/vis_dataset.py --debugging --cfg configs/cotton_ft_weakaugs.yaml --seed 1 --batch_size 3 --serial 421 --vis_cols 3 --vis_class 0
python -u tools/vis_dataset.py --debugging --cfg configs/soyageing_ft_weakaugs.yaml --seed 1 --batch_size 3 --serial 421 --vis_cols 3 --vis_class 0
python -u tools/vis_dataset.py --debugging --cfg configs/soyglobal_ft_weakaugs.yaml --seed 1 --batch_size 3 --serial 421 --vis_cols 3 --vis_class 0
python -u tools/vis_dataset.py --debugging --cfg configs/soygene_ft_weakaugs.yaml --seed 1 --batch_size 3 --serial 421 --vis_cols 3 --vis_class 0
python -u tools/vis_dataset.py --debugging --cfg configs/soylocal_ft_weakaugs.yaml --seed 1 --batch_size 3 --serial 421 --vis_cols 3 --vis_class 0

python -u tools/vis_dataset.py --debugging --cfg configs/cotton_ft_weakaugs.yaml --seed 1 --batch_size 3 --serial 421 --vis_cols 3 --vis_class 1
python -u tools/vis_dataset.py --debugging --cfg configs/soyageing_ft_weakaugs.yaml --seed 1 --batch_size 3 --serial 421 --vis_cols 3 --vis_class 1
python -u tools/vis_dataset.py --debugging --cfg configs/soyglobal_ft_weakaugs.yaml --seed 1 --batch_size 3 --serial 421 --vis_cols 3 --vis_class 1
python -u tools/vis_dataset.py --debugging --cfg configs/soygene_ft_weakaugs.yaml --seed 1 --batch_size 3 --serial 421 --vis_cols 3 --vis_class 1
python -u tools/vis_dataset.py --debugging --cfg configs/soylocal_ft_weakaugs.yaml --seed 1 --batch_size 3 --serial 421 --vis_cols 3 --vis_class 1

