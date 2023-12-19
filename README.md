# Learning to Manipulate Artistic Images (AAAI 2024)

## Abstract
Recent advancement in computer vision has significantly lowered the barriers to artistic creation. Exemplar-based image translation methods have attracted much attention due to flexibility and controllability. However, these methods hold assumptions regarding semantics or require semantic information as the input, while accurate semantics is not easy to obtain in artistic images. Besides, these methods suffer from cross-domain artifacts due to training data prior and generate imprecise structure due to feature compression in the spatial domain. In this paper, we propose an arbitrary Style Image Manipulation Network (SIM-Net), which leverages semantic-free low level information as guidance and a \textit{region transportation} strategy for image generation. Our method balances computational efficiency and high resolution to a certain extent. Moreover, our method facilitates zero-shot style image manipulation. Both qualitative and quantitative experiments demonstrate the superiority of our method over state-of-the-art methods.

## Demo

<!-- ![Demo](imgs/demo.gif) -->
<p align="center">
  <img src="./demo.gif">
</p>

## Installation

Clone this repo.
```bash
git clone https://github.com/SnailForce/SIM-Net.git
cd SIM-Net
``` 

Clone the Synchronized-BatchNorm-PyTorch repository.
```bash
cd models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

This code requires PyTorch, python 3+. Please install dependencies by
``` bash
pip install -r requirements.txt
```
We recommend to install Pytorch version after `Pytorch 1.8.0`.

## Inference Using Pretrained Model

Download the pretrained model from [here](https://drive.google.com/drive/folders/1gGKlXERnRMUcAQXtN5fJhmRcy3jyNLgL?usp=sharing) and save them in checkpoints/SIM-Net. Then run the command
``` bash
python test.py 
       --checkpoints_dir ./checkpoints/ 
       --dataroot your_dataroot
       --name SIM-Net 
       --model cycle_gan 
       --results_dir results 
       --epoch latest 
       --config config_channel.yaml 
       --dataset_mode test
```
Note that `--dataroot` parameter is your dataset root, e.g. `datasets/ade20k`.

## Training

Pretrained VGG model move it to `models/`. This model is used to calculate training loss.

Examples of training data are in the folder `datasets/all_in`
subdirectories under the `datasets/all_in/class` folder are distinguished according to the class of the training data (`building` and `face` is used as an example in the file) 
`datasets/all_in/config_channel.yaml` defines details of networks.

```bash
python train.py 
       --checkpoints_dir ./checkpoints/ 
       --dataroot
       --batch_size 32 
       --print_freq 120 
       --save_epoch_freq 1 
       --display_freq 80 
       --display_ncols 6 
       --display_id 1 
       --name SIM-Net 
       --model cycle_gan
       --gpu_ids 0,1,2,3,4,5,6,7 
       --no_flip 
       --load_size 256 
       --display_port 8097 
       --save_latest_freq 1200 
       --dataset_mode mask 
       --n_epochs 4  
       --n_epochs_decay 2  
       --max_dataset_size 5000 
       --config config_channel.yaml  
       --init_type normal
```

Note that `--dataroot` parameter is your dataset root, e.g. `datasets/ade20k`.
You can set `batchsize` to 16, 8 or 4 with fewer GPUs and change `gpu_ids`.

To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

Place the test images in the `imgs/test_data` folder and name the file with reference to the example given. 
The `.jpg` and `.png` files in the `imgs/test_data/style` folder represent the input Exemplar with its corresponding source ROIs, and the `.png` file with the same name in the `imgs/test_data/content` folder represents its driving ROIs.

In addition, since our method does not care about specific semantic information, for labelmap, only different values from 0~255 need to be used to represent different ROIs respectively, and the correspondence needs to be maintained in source and driving, while the choice of which specific value does not affect the final result.

## Dataset
Will bre released soon.

## License
All rights reserved. Licensed under the MIT License.
The code is released for academic research use only.

## Citation

if you use this code for your research, please cite our paper.

```
@InProceedings{Guo2024learning
  author    = {Wei Guo, Yuqi Zhang, De Ma, Qian Zheng},
  title     = {Learning to Manipulate Artistic Images},
  booktitle = {Proceedings of the AAAI conference on artificial intelligence},
  year      = {2024}
}
```

## Acknowledgments
The code borrows from [segment anything](https://github.com/facebookresearch/segment-anything), [first-order-model](https://github.com/AliaksandrSiarohin/first-order-model), [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch).
The demo code borrows from [SEAN](https://github.com/ZPdesu/SEAN) and will be released soon.

