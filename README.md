# SPME-UNet

## 1. Download pre-trained model (TransNeXt-B)
* [Get pre-trained model in this link] (https://github.com/DaiShiResearch/TransNeXt/tree/main): Put pretrained model into folder "pretrained_ckpt/"

## 2. Prepare

```bash
cd swattention_extension
pip install .
```

## 3. Environment

- Please prepare an environment with python=3.7, and then use the command 
```bash
pip install -r requirements.txt
```
for the dependencies.

## 4. Train/Test

- Run the train script on synapse dataset.  If you do not have enough GPU memory, the bacth size can be reduced to 12 or 6 to save memory.

- Train

```bash
python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24
```

- Test 

```bash
python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_savenii --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```


