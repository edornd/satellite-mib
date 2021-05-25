# Modeling the Background for Incremental Learning in Semantic Segmentation
## Adaptation to multispectral aerial imagery

This is an adaptation of MiB to different domains, for more information please check [the original implementation](https://github.com/fcdl94/MiB/).

# Installation notes
1. make sure you have GCC and G++ 7
2. clone apex, if not present already, and launch the following command

```bash
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

3. install other requirements
4. launch mib:

```bash
python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root $DATA_ROOT --batch_size 12 --dataset ade --name MiB --task 100-50 --step 0 --lr 0.001 --epochs 60 --method MiB
python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root $DATA_ROOT --batch_size 12 --dataset voc --name MiB --task 100-50 --step 1 --lr 0.001 --epochs 60 --method MiB
python -m torch.distributed.launch --nproc_per_node=1 run.py --data_root $DATA_ROOT --batch_size 12 --dataset voc --name MiB --task 15-5 --step 1 --lr 0.001 --epochs 60 --method MiB
```
