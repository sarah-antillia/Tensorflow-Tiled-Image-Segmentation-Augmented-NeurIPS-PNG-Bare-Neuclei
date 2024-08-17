<h2>Tensorflow-Tiled-Image-Segmentation-Augmented-NeurIPS-PNG-Bare-Nuclei (2024/08/17)</h2>

This is the second experiment of Tiled Image Segmentation for <a href="https://zenodo.org/records/10719375">NeurIPS 2022 CellSegmentation</a>
 based on
the <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/file/d/1q1cDVdAqOPRhVegDUKYgxbRqBvVVnqB2/view?usp=sharing">
Tiled-Cell-ImageMask-Dataset-PNG-Bare-Nuclei</a>, which was derived by us from the datset of zenodo.org website: 
<a href="https://zenodo.org/records/10719375">NeurIPS 2022 CellSegmentation</a>
<br>
<br>
Please see also the first experiment 
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-NeurIPS-Cell">Tensorflow-Tiled-Image-Segmentation-Augmented-NeurIPS-Cell</a>
<br><br>
We generated the Tiled-PNG-Bare-Nuclei dataset from 86 Brightfield Bare Nuclei png image files and corresponding label files in  
<a href="https://zenodo.org/records/10719375/files/Training-labeled.zip?download=1">
<b>Training-labeled</b></a> dataset in <a href="https://zenodo.org/records/10719375">NeurIPS 2022 CellSegmentation</a>
<br> 
<br>

In this experiment, we employed the following strategy:
<b>
<br>
1. We trained and validated a TensorFlow UNet model using the Tiled-Cell-ImageMask-Dataset for Bare-Neuclei, 
which was tiledly-splitted to 512x512
 and reduced to 512x512 image and mask dataset.<br>
2. We applied the Tiled-Image Segmentation inference method to predict the neuclei regions for the mini_test images 
with a resolution of 2K and 4K pixels. 
<br><br>
</b>  
Please note that Tiled-Cell-ImageMask contains two types of images and masks:<br>
1. Tiledly-splitted to 512x512 image and mask files.<br>
2. Size-reduced to 512x512 image and mask files.<br>
Namely, this is a mixed set of Tiled and Non-Tiled ImageMask Datasets.

<hr>
<b>Actual Tiled Image Segmentation for Images of 2090x2090, 2090x2090 and 4096x4096 pixels</b><br>
As shown below, the tiled inferred masks look similar to the ground truth masks.
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: tiled inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/images/10051.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/masks/10051.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test_output_tiled/10051.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/images/10065.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/masks/10065.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test_output_tiled/10065.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/images/10085.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/masks/10085.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test_output_tiled/10085.jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we have used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Cell Segmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
The original dataset used here has been taken from the following website:<br> 

<a href="https://neurips22-cellseg.grand-challenge.org/dataset/"><b>Weakly Supervised Cell Segmentation in Multi-modality 
High-Resolution Miscroscopy Images</b>
</a>
<br><br>
<b>Download dataset</b><br>
You can download a training dataset corresponding to the cell segmentation from the zendo.org website:  
<a href="https://zenodo.org/records/10719375/files/Training-labeled.zip?download=1">Training-labeled.zip
</a><br>
<br>
<b>NeurIPS 2022 Cell Segmentation Competition Dataset</b><br>
Ma, Jun, Xie, Ronald, Ayyadhury, Shamini, Ge, Chen, Gupta, Anubha, Gupta, Ritu, Gu, Song, <br>
Zhang, Yao, Lee, Gihun, Kim, Joonkee, Lou, Wei, Li, Haofeng, Upschulte, Eric, Dickscheid, Timo,<br>
de Almeida, José Guilherme, Wang, Yixin, Han, Lin,Yang, Xin, Labagnara, Marco,Gligorovski, Vojislav,<br>
Scheder, Maxime, Rahi, Sahand Jamal,Kempster, Carly, Pollitt, Alice, Espinosa, Leon, Mignot, Tam,<br>
Middeke, Jan Moritz, Eckardt, Jan-Niklas, Li, Wangkai, Li, Zhaoyang, Cai, Xiaochen, Bai, Bizhe,<br>
Greenwald, Noah F., Van Valen, David, Weisbart, Erin, Cimini, Beth A, Cheung, Trevor, Brück, Oscar,<br>
Bader, Gary D.,Wang, Bo<br>

Zenodo. https://doi.org/10.5281/zenodo.10719375<br>
<br> 
<b>Data license</b>: CC BY-NC-ND
<br>

<h3>
<a id="2">
2 Tiled-Cell ImageMask Dataset
</a>
</h3>
 If you would like to train this Tiled-Cell Segmentation model by yourself,
 please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/1q1cDVdAqOPRhVegDUKYgxbRqBvVVnqB2/view?usp=sharing">
Tiled-Cell-ImageMask-Dataset-PNG-Bare-Nuclei</a>,
<br>
expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Tiled-Cell
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>

<b>Tiled-Cell Dataset Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/Tiled-Cell-ImageMask-Dataset-PNG-Bare-Nuclei_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not so large to use for a training set for our segmentation model, 
so we used an online augmentation tool <a href="./src/ImageMaskAugmentor.py">ImageMaskAugmentor.py</a> to increase the number of the dataset 
during the training process. 
<br>

<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained Tiled-Cell TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/Tiled-Cell and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>
<pre>
; train_eval_infer.config
; 2024/08/16 (C) antillia.com

[model]
model          = "TensorflowUNet"
generator      = True
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = False
normalization  = False
num_classes    = 1
base_filters   = 16
base_kernels   = (7,7)
num_layers     = 8
dropout_rate   = 0.05
learning_rate  = 0.0001
clipvalue      = 0.5
dilation       = (1,1)
;loss           = "bce_iou_loss"
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
show_summary   = False

[train]
epochs        = 100
batch_size    = 2
steps_per_epoch  = 240
validation_steps = 80
patience      = 10

;metrics       = ["iou_coef", "val_iou_coef"]
metrics       = ["dice_coef", "val_dice_coef"]

model_dir     = "./models"
eval_dir      = "./eval"
image_datapath = "../../../dataset/Tiled-Cell/train/images/"
mask_datapath  = "../../../dataset/Tiled-Cell/train/masks/"

epoch_change_infer     = True
epoch_change_infer_dir = "./epoch_change_infer"
epoch_change_tiledinfer     = True
epoch_change_tiledinfer_dir = "./epoch_change_tiledinfer"
num_infer_images       = 1

create_backup  = False

learning_rate_reducer = True
reducer_factor     = 0.3
reducer_patience   = 4
save_weights_only  = True

[eval]
image_datapath = "../../../dataset/Tiled-Cell/valid/images/"
mask_datapath  = "../../../dataset/Tiled-Cell/valid/masks/"

[test] 
image_datapath = "../../../dataset/Tiled-Cell/test/images/"
mask_datapath  = "../../../dataset/Tiled-Cell/test/masks/"

[infer] 
images_dir    = "./mini_test/images"
output_dir    = "./mini_test_output"

[tiledinfer] 
overlapping   = 64
images_dir    = "./mini_test/images"
output_dir    = "./mini_test_output_tiled"

[segmentation]
colorize      = True
black         = "black"
white         = "green"
blursize      = None

[image]
;color_converter = None
color_converter = "cv2.COLOR_BGR2HSV_FULL"
gamma           = 0

[mask]
blur      = False
blur_size = (3,3)
binarize  = False
;threshold = 128
threshold = 80

[generator]
debug        = False
augmentation = True

[augmentor]
vflip    = True
hflip    = True
rotation = True
angles   = [90, 180, 210, 270, ]
shrinks  = [0.8]
shears   = [0.1]

deformation = True
distortion  = True
sharpening  = False
brightening = False
barrdistortion = True

[deformation]
alpah     = 1300
sigmoids  = [8.0,]

[distortion]
gaussian_filter_rsigma= 40
gaussian_filter_sigma = 0.5
distortions           = [0.02,]

[barrdistortion]
radius = 0.3
amount = 0.3
centers =  [(0.3, 0.3), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.7, 0.7)]

[sharpening]
k        = 1.0

[brightening]
alpha  = 1.2
beta   = 10  
</pre>
<hr>
<b>Model parameters</b><br>
Defined small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16 
base_kernels   = (7,7)
num_layers     = 8
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate   = 0.0001
</pre>

<b>Online augmentation</b><br>
Enabled our online augmentation.  
<pre>
[model]
model         = "TensorflowUNet"
generator     = True
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.3
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer and epoch_change_tiledinfer callbacks.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_change_tiledinfer  = True
epoch_change_tiledinfer_dir = "./epoch_change_tiledinfer"
num_infer_images         = 1
</pre>

By using these callbacks, on every epoch_change, the inference procedures can be called
 for an image in <b>mini_test</b> folder. These will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/asset/epoch_change_infer.png" width="1024" height="auto"><br>
<br>
<br>
<b>Epoch_change_tiled_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/asset/epoch_change_tiledinfer.png" width="1024" height="auto"><br>
<br>
<br>
In this experiment, the training process was stopped at epoch 28 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/asset/train_console_output_at_epoch_28.png" width="720" height="auto"><br>
<br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Tiled-Cell.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/asset/evaluate_console_output_at_epoch_28.png" width="720" height="auto">
<br><br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) score to this <a href="./projects/dataset/Tiled-Cell/test/">Tiled-Cell/test</a> was relatively low, but dice_coef not so high.
<pre>
loss,0.1136
dice_coef,0.8121
</pre>


<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Tiled-Cell.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>


<h3>
6 Tiled Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Tiled-Cell.<br>
<pre>
./4.tiled_infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetTiledInferencer.py ./train_eval_infer_aug.config
</pre>

<hr>
<b>Tiled inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/asset/mini_test_output_tiledinfer.png" width="1024" height="auto"><br>
<br>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Tiled-inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/images/10056.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/masks/10056.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test_output_tiled/10056.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/images/10058.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/masks/10058.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test_output_tiled/10058.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/images/10075.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/masks/10075.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test_output_tiled/10075.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/images/10082.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/masks/10082.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test_output_tiled/10082.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/images/10086.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/masks/10086.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test_output_tiled/10086.jpg" width="320" height="auto"></td>
</tr>
</table>

<br>
<br>
<!--
  -->
<b>Comparison of Non-tiled inferred mask and Tiled-Inferred mask</b><br>
As shown below, the tiled inferencer based on our simple UNet model can generate far better results than non-tiled inferencer.
<br>
<table>
<tr>
<th>Mask (ground_truth)</th>

<th>Non-tiled-inferred-mask</th>
<th>Tiled-inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/masks/10051.jpg" width="320" height="auto"></td>

<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test_output/10051.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test_output_tiled/10051.jpg" width="320" height="auto"></td>

</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/masks/10055.jpg" width="320" height="auto"></td>

<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test_output/10055.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test_output_tiled/10055.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test/masks/10085.jpg" width="320" height="auto"></td>

<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test_output/10085.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-Cell/mini_test_output_tiled/10085.jpg" width="320" height="auto"></td>
</tr>
</table>
<br>

<h3>
References
</h3>
<b>1. Multi-stream Cell Segmentation with Low-level Cues for Multi-modality Images</b><br>
Wei Lou, Xinyi Yu, Chenyu Liu , Xiang Wan, Guanbin Li, Siqi Liu, Haofeng Li<br>
<a href="https://arxiv.org/pdf/2310.14226">https://arxiv.org/pdf/2310.14226</a>
<br>
<br>

<b>2. MEDIAR: Harmony of Data-Centric and Model-Centric for Multi-Modality Microscopy
</b><br>
Lee-Gihun <br>
<a href="https://github.com/Lee-Gihun/MEDIAR">https://github.com/Lee-Gihun/MEDIAR</a>
<br>
<br>
<b>3. NeurIPS-CellSeg
</b><br>
JunMa11 <br>
<a href="https://github.com/JunMa11/NeurIPS-CellSeg">https://github.com/JunMa11/NeurIPS-CellSeg</a>
<br>

<br>
<b>4. Tensorflow-Tiled-Image-Segmentation-Augmented-NeurIPS-Cell
</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-NeurIPS-Cell">
https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-NeurIPS-Cell</a>
<br>
