# How does the machine perceive depth for indoor single images?
<!--  ## Depth Insight - Contribution of Different Features to Indoor Single-image Depth Estimation -->

 
## Appendix for ICIP

[Here](/Appendix/ICIP_Appendix.pdf) is the Appendix for ICIP submission. Please have a check. 

**Abstract**

Depth estimation from a single image is a challenging problem in computer vision because binocular disparity or motion information is absent. Whereas impressive performances have been reported in this area recently using end-to-end trained deep neural architectures, as to what cues in the images are being exploited by these black box systems is hard to know. To this end, in this work, we quantify the relative contributions of the known cues of depth in a monocular depth estimation setting using an indoor scene data set. Our work uses feature extraction techniques to relate the single features of shape, texture, colour and saturation, taken in isolation, to predict depth. We find that the shape of objects extracted by edge detection substantially contributes more than others in the indoor setting considered, while the other features also have contributions in varying degrees. These insights will help optimise depth estimation models, boosting their accuracy and robustness. They promise to broaden the practical applications of vision-based depth estimation. 


## Data
The NYU-Depth V2 dataset includes video sequences from different indoor environments, captured using the RGB and Depth cameras of the Microsoft Kinect. The dataset is available for download at the following link:
[NYU Dataset](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2)

## Pre-trained Models
The pre-trained models can be downloaded at [here](https://drive.google.com/drive/folders/1N5jkP80hznZplgmMf_u2P0fmopg-ZmOJ?usp=drive_link)  (It seems to suddenly fail sometimes. Please drop an email if it happens)

## Environment
Configure the environment according to the [environment configuration file](/config/DI_environment.txt).

## Pre-processing
### Phase Scrambling
Colour, grayscale and saturation features can be generated from original images by using the code available [here](/tools/phaseScrambling). 
Texture feature can be generated using the code [here](/tools/shuffle).

## Inference
Please configure the following three parameters: the path of the pre-trained model, the type of model to be used, and the path of the test dataset. For a fair comparison, inputs with a single channel will be duplicated into three channels and fed into the same model.
```python
--trained_model # choose pre-trained model in /pretrainedModel
--featureType # 'rgb', 'single_grayscale', 'pseudo_rgb', 'shape2'
--filenames_file_eval # data list 
```
- For cases using original RGB images as input:
  - Use the original data list.
  - Set `featureType` to 'rgb'.
- For colour:
  - Generate a new dataset and obtain its data list during pre-processing.
  - Use the corresponding trained_model.
  - Set `featureType` to 'rgb'.
- For grayscale features:
  - Use the same data list as for colour features.
  - Set `featureType` to 'single_grayscale'.
- For saturation and texture features:
  - Generate a new dataset and obtain its data list during pre-processing.
  - Use the corresponding trained_model.
  - Set `featureType` to 'pseudo_rgb'.
- For shape features:
  - Use the original data list.
  - Set `featureType` to 'shape2'.
  
  
## Recover
For the results after phase scrambling, use the [corresponding method](/tools/phaseScrambling/recover_color.py) to restore the positions of objects in the outputs, based on the pre-saved random matrix.

For the results after shuffling, use the [corresponding method](/tools/shuffle/recover.py) to restore the positions of objects in the outputs, based on the pre-saved random list.

## Reference
```
@article{couprie2013indoor,
  title={Indoor semantic segmentation using depth information},
  author={Couprie, Camille and Farabet, Cl{\'e}ment and Najman, Laurent and LeCun, Yann},
  journal={arXiv preprint arXiv:1301.3572},
  year={2013}
}

@article{alhashim2018high,
  title={High quality monocular depth estimation via transfer learning},
  author={Alhashim, Ibraheem and Wonka, Peter},
  journal={arXiv preprint arXiv:1812.11941},
  year={2018}
}

@inproceedings{bhat2021adabins,
  title={Adabins: Depth estimation using adaptive bins},
  author={Bhat, Shariq Farooq and Alhashim, Ibraheem and Wonka, Peter},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4009--4018},
  year={2021}
}

@inproceedings{ge2022contributions,
  title={Contributions of shape, texture, and color in visual recognition},
  author={Ge, Yunhao and Xiao, Yao and Xu, Zhi and Wang, Xingrui and Itti, Laurent},
  booktitle={European Conference on Computer Vision},
  pages={369--386},
  year={2022},
  organization={Springer}
}
```
  
