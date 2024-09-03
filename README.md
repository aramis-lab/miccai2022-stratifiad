# MICCAI 2022 - Paper ID: 2116

## How to cite the paper:
If you use this code and/or the dataset, please cite the following paper:
```
@InProceedings{miccai-paper-2116,
author="Jimenez, Gabriel and Kar, Anuradha and Ounissi, Mehdi and Ingrassia, Léa and Boluda, Susana and Delatour, Benoît and Stimmer, Lev and Racoceanu, Lev",
title="Visual DL-based explanation for neuritic plaques segmentation in Alzheimer's Disease",
booktitle="Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2022",
year="2022",
publisher="Springer International Publishing",
}
```

## Setting up the OS:
- This code was develop and runs properly in Ubuntu 18.04 and with Python 3.6.9

```
sudo apt update
```

- In case `pip3` is not installed, then run:

```shell
sudo apt install python3-pip
```

- Install `pipenv`:

```shell
pip3 install --user pipenv
```

## Setting up the environment and dependencies:
- Inside the main directory, run the following command to syncronize all the dependencies:

```shell
pipenv sync
```

- To activate the virtual environment:

```shell
pipenv shell
```

## Structure of the dataset:
The acquisition of the WSI and the patch generation process is described in the article. The dataset used for the three experiments below can be found [here](https://doi.org/10.6084/m9.figshare.20188142.v1) (size: 128x128) and [here](https://doi.org/10.6084/m9.figshare.20186840.v1) (size: 256x256). The structure is as follows:

```
dataset
├── 128x128
│   └── new_wsi_00
│   │   └── macenko
│   │   └── masks
│   │   └── patches
│   │   └── vahadane
│   └── new_wsi_01
│   │   └── macenko
│   │   └── masks
│   │   └── patches
│   │   └── vahadane
│   └── ...
└── 256x256
    └── new_wsi_00
    │   └── macenko
    │   └── masks
    │   └── patches
    └── new_wsi_01
    │   └── macenko
    │   └── masks
    │   └── patches
    └── ...
```

|  **folder**  | **description** |
|:------------:|:---------------:|
|  **macenko** | contains all the patches and its corresponding corner augmentations using Macenko method for color normalization. Files are named in sequential order according to the annotations in the WSI and the augmented ones have an additional `_Cx.png` added to the name, where `x` is one of the four corners. |
|    **masks** | contains all the masks and its corresponding corner augmentations. Files are named in sequential order according to the annotations in the WSI and the augmented ones have an additional `_Cx.png` added to the name, where `x` is one of the four corners. |
|    **patches** | contains all the original unnormalized patches and its corresponding corner augmentations. Files are named in sequential order according to the annotations in the WSI and the augmented ones have an additional `_Cx.png` added to the name, where `x` is one of the four corners. |
| **vahadane** | contains all the patches and its corresponding corner augmentations using Vahadane method for color normalization. Files are named in sequential order according to the annotations in the WSI and the augmented ones have an additional `_Cx.png` added to the name, where `x` is one of the four corners. Only available for 128x128. |

## Running the experiments: 
### **UNet - Experiment 01:**
This experiment uses the 8 WSI images in the dataset and the Macenko color normalization method. We use this experiment to analyze how the network performs under different modalities and also when increasing the environment of the object (i.e., different patch sizes).

- For the 128x128 patch-size dataset, with cross validation and cross testing:
```shell
cd unet
sh test_00_128x128.sh
sh test_01_128x128.sh
sh test_02_128x128.sh
sh test_03_128x128.sh
```

- For the 256x256 patch-size dataset, with cross validation and cross testing:
```shell
cd unet
sh test_00_256x256.sh
sh test_01_256x256.sh
sh test_02_256x256.sh
sh test_03_256x256.sh
```

### **UNet - Experiment 02:**
This experiment evaluates the impact of the scanner in the segmentation of plaques. We use only the 128x128 patch-size dataset as we obtained better performance with this patches in the first experiment.

- To train using the WSI from the scanner Hamamatsu NanoZoomer 2.0-RS:
```shell
cd unet
sh test_00_128x128_oldscan.sh
```

- To train using the WSI from the scanner Hamamatsu NanoZoomer S60:
```shell
cd unet
sh test_00_128x128_newscan.sh
```

### **UNet - Experiment 03:**
This experiment evaluates the impact of the color normalization. We use only the 128x128 patch-size dataset and the configuration of the best fold from the first experiment.

```shell
cd unet
sh test_00_128x128_bestfold_vahadane.sh
```

### **Attention UNet:**
This experiment evaluates the performance of the attention UNet in different patch size datasets. You will need to manually configure the following line in the file `train_att_unet.py` (provided that your data follows the structure mentioned above):

- To use the 128x128 dataset:
```python
main_data_dir = os.path.join('..', '..', '..','dataset','128x128')
```

- To use the 256x256 dataset:
```python
main_data_dir = os.path.join('..', '..', '..','dataset','256x256')
```

- To run the experiment:
```shell
python train_att_unet.py
```

## Full results:
### **UNet and Attention UNet:**

Table 01. Architecture used: UNet. Patch size: 128x128 pixels (best fold is reported in bold font).
|   **fold_name**   | **dev_dice** | **dev_f1** | **dev_recall** | **dev_precision** | **test_dice** | **test_f1** | **test_recall** | **test_precision** |
|:-----------------:|:------------:|:----------:|:--------------:|:-----------------:|:-------------:|:-----------:|:---------------:|:------------------:|
|   test_00_cv_00   |    0.7151    |   0.7165   |     0.6674     |       0.8017      |     0.6753    |    0.6707   |      0.668      |        0.784       |
|   test_00_cv_01   |    0.7034    |   0.6933   |      0.699     |       0.718       |     0.7046    |    0.7035   |      0.7495     |       0.7475       |
|   test_00_cv_02   |    0.6963    |   0.6932   |     0.6873     |       0.7339      |     0.6781    |    0.6765   |      0.7423     |       0.7094       |
|   test_01_cv_00   |    0.7011    |   0.7037   |      0.714     |       0.7239      |     0.7032    |    0.6962   |      0.6684     |       0.8052       |
| **test_01_cv_01** |  **0.7231**  | **0.7118** |   **0.6763**   |     **0.7801**    |   **0.7248**  |  **0.7192** |    **0.7105**   |     **0.8067**     |
|   test_01_cv_02   |     0.72     |   0.7217   |     0.7519     |       0.7185      |     0.7141    |    0.7068   |      0.6811     |       0.8166       |
|   test_02_cv_00   |     0.709    |   0.7156   |     0.7195     |       0.7423      |     0.7027    |    0.7004   |      0.7043     |       0.7855       |
|   test_02_cv_01   |    0.7127    |   0.7195   |     0.7012     |       0.7618      |     0.6643    |    0.6608   |      0.6444     |       0.7996       |
|   test_02_cv_02   |    0.6807    |   0.6838   |     0.6862     |       0.7167      |     0.6306    |    0.6296   |      0.6634     |       0.7316       |
|   test_03_cv_00   |    0.6765    |   0.6813   |     0.6788     |       0.7183      |     0.6845    |    0.6855   |      0.8061     |        0.666       |
|   test_03_cv_01   |    0.6959    |   0.6967   |     0.6244     |       0.8167      |     0.6883    |    0.879    |      0.7981     |       0.6777       |
|   test_03_cv_02   |    0.6112    |   0.6008   |      0.61      |       0.6325      |     0.6521    |    0.6545   |      0.8018     |       0.6245       |
|      **mean**     |    0.6954    |   0.6948   |     0.6847     |       0.7387      |     0.6852    |    0.6986   |      0.7198     |       0.7462       |
|      **std**      |    0.0289    |   0.0313   |     0.0373     |       0.0462      |     0.0260    |    0.0596   |      0.0561     |       0.0615       |
|      **max**      |    0.7231    |   0.7217   |     0.7519     |       0.8167      |     0.7248    |    0.879    |      0.8061     |       0.8166       |
|      **min**      |    0.6112    |   0.6008   |      0.61      |       0.6325      |     0.6306    |    0.6296   |      0.6444     |       0.6245       |

Table 02. Architecture used: Attention UNet. Patch size: 128x128 pixels (best fold is reported in bold font).
|   **fold_name**   | **dev_dice** | **dev_f1** | **dev_recall** | **dev_precision** | **test_dice** | **test_f1** | **test_recall** | **test_precision** |
|:-----------------:|:------------:|:----------:|:--------------:|:-----------------:|:-------------:|:-----------:|:---------------:|:------------------:|
|   test_00_cv_00   |    0.7654    |   0.7654   |     0.7833     |       0.8000      |     0.6405    |    0.6405   |      0.5959     |       0.8177       |
|   test_00_cv_01   |    0.7216    |   0.7216   |     0.7293     |       0.7890      |     0.6734    |    0.6734   |      0.6387     |       0.8258       |
|   test_00_cv_02   |    0.7428    |   0.7422   |     0.7425     |       0.8108      |     0.6499    |    0.6499   |      0.6320     |       0.7883       |
|   test_01_cv_00   |    0.7646    |   0.7646   |     0.7876     |       0.7934      |     0.6851    |    0.6851   |      0.6582     |       0.7979       |
|   test_01_cv_01   |    0.7272    |   0.7272   |     0.7916     |       0.7299      |     0.7128    |    0.7122   |      0.7454     |       0.7522       |
|   test_01_cv_02   |    0.7335    |   0.7335   |     0.7593     |       0.7772      |     0.6909    |    0.6909   |      0.6836     |       0.7890       |
|   test_02_cv_00   |    0.7658    |   0.7658   |     0.7946     |       0.7900      |     0.7184    |    0.7184   |      0.7109     |       0.8060       |
| **test_02_cv_01** |  **0.7159**  | **0.7153** |   **0.7565**   |     **0.7429**    |   **0.7263**  |  **0.7263** |    **0.7950**   |     **0.7254**     |
|   test_02_cv_02   |    0.7809    |   0.7809   |     0.8232     |       0.7855      |     0.6948    |    0.6948   |      0.7241     |       0.7488       |
|   test_03_cv_00   |    0.8193    |   0.8193   |     0.8180     |       0.8559      |     0.6959    |    0.6959   |      0.6839     |       0.8008       |
|   test_03_cv_01   |    0.6962    |   0.6962   |     0.6843     |       0.7908      |     0.7140    |    0.7140   |      0.7781     |       0.7281       |
|   test_03_cv_02   |    0.7857    |   0.7857   |     0.8297     |       0.7896      |     0.7024    |    0.7024   |      0.8435     |       0.6571       |
|      **mean**     |    0.7516    |   0.7515   |     0.7750     |       0.7879      |     0.6920    |    0.6920   |      0.7074     |       0.7698       |
|      **std**      |    0.0334    |   0.0335   |     0.0408     |       0.0301      |     0.0254    |    0.0254   |      0.0703     |       0.0469       |
|      **max**      |    0.8193    |   0.8193   |     0.8297     |       0.8559      |     0.7263    |    0.7263   |      0.8435     |       0.8258       |
|      **min**      |    0.6962    |   0.6962   |     0.6843     |       0.7299      |     0.6405    |    0.6405   |      0.5959     |       0.6571       |


Table 03. Architecture used: UNet. Patch size: 256x256 pixels (best fold is reported in bold font).
|   **fold_name**   | **dev_dice** | **dev_f1** | **dev_recall** | **dev_precision** | **test_dice** | **test_f1** | **test_recall** | **test_precision** |
|:-----------------:|:------------:|:----------:|:--------------:|:-----------------:|:-------------:|:-----------:|:---------------:|:------------------:|
|   test_00_cv_00   |     0.707    |   0.7095   |     0.6557     |       0.7962      |     0.6423    |    0.6363   |      0.5839     |       0.8071       |
|   test_00_cv_01   |    0.6509    |   0.6563   |     0.6744     |       0.6604      |     0.6783    |    0.6743   |      0.6563     |       0.7767       |
|   test_00_cv_02   |    0.6663    |   0.6657   |     0.6725     |       0.6826      |     0.6368    |    0.6322   |      0.6356     |       0.7269       |
| **test_01_cv_00** |  **0.6939**  | **0.7099** |   **0.7041**   |     **0.7319**    |   **0.6963**  |  **0.6893** |    **0.6632**   |     **0.7998**     |
|   test_01_cv_01   |    0.6411    |    0.642   |     0.6018     |       0.7157      |     0.6856    |    0.6785   |      0.6468     |       0.8008       |
|   test_01_cv_02   |    0.6793    |   0.6829   |     0.6892     |       0.6979      |     0.6718    |    0.6654   |      0.6487     |       0.7732       |
|   test_02_cv_00   |    0.6893    |   0.7077   |      0.716     |       0.716       |     0.6395    |    0.6352   |      0.6104     |        0.766       |
|   test_02_cv_01   |    0.6911    |    0.698   |     0.6344     |       0.8027      |     0.621     |    0.6155   |      0.5748     |       0.7816       |
|   test_02_cv_02   |    0.6462    |    0.649   |     0.6593     |       0.6638      |     0.5949    |    0.595    |      0.6676     |       0.6185       |
|   test_03_cv_00   |    0.6432    |   0.6529   |     0.6323     |       0.7029      |     0.6507    |     0.65    |      0.7633     |       0.6242       |
|   test_03_cv_01   |    0.6747    |   0.6877   |     0.6581     |       0.7453      |     0.6507    |    0.6521   |      0.8146     |       0.5945       |
|   test_03_cv_02   |    0.5391    |   0.5355   |     0.5558     |       0.5535      |     0.5834    |    0.5856   |      0.8118     |       0.5149       |
|      **mean**     |     0.660    |    0.666   |      0.654     |       0.706       |     0.646     |    0.642    |      0.673      |        0.715       |
|      **std**      |     0.042    |    0.046   |      0.042     |       0.063       |     0.033     |    0.031    |      0.077      |        0.096       |
|      **max**      |     0.707    |    0.710   |      0.716     |       0.803       |     0.696     |    0.689    |      0.815      |        0.807       |
|      **min**      |    0.5391    |   0.5355   |     0.5558     |       0.5535      |     0.5834    |    0.5856   |      0.5748     |       0.5149       |

Table 04. Architecture used: Attention UNet. Patch size: 256x256 pixels (best fold is reported in bold font).
|   **fold_name**   | **dev_dice** | **dev_f1** | **dev_recall** | **dev_precision** | **test_dice** | **test_f1** | **test_recall** | **test_precision** |
|:-----------------:|:------------:|:----------:|:--------------:|:-----------------:|:-------------:|:-----------:|:---------------:|:------------------:|
|   test_00_cv_00   |    0.7288    |   0.7288   |     0.8318     |       0.6819      |     0.6439    |    0.6439   |      0.6312     |       0.7429       |
|   test_00_cv_01   |    0.6698    |   0.6698   |     0.7152     |       0.6855      |     0.6038    |    0.6038   |      0.5860     |       0.7462       |
|   test_00_cv_02   |    0.6644    |   0.6644   |     0.6396     |       0.7920      |     0.5638    |    0.5638   |      0.5132     |       0.7701       |
|   test_01_cv_00   |    0.7349    |   0.7349   |     0.8246     |       0.6971      |     0.6560    |    0.6560   |      0.6724     |       0.7162       |
| **test_01_cv_01** |  **0.6471**  | **0.6471** |   **0.6618**   |     **0.7102**    |   **0.6796**  |  **0.6790** |    **0.6746**   |     **0.7599**     |
|   test_01_cv_02   |    0.6774    |   0.6774   |     0.6539     |       0.7842      |     0.6307    |    0.6307   |      0.5998     |       0.7847       |
|   test_02_cv_00   |    0.7463    |   0.7463   |     0.8221     |       0.7172      |     0.6323    |    0.6323   |      0.6289     |       0.7087       |
|   test_02_cv_01   |    0.6767    |   0.6761   |     0.7026     |       0.7199      |     0.6497    |    0.6497   |      0.6715     |       0.6983       |
|   test_02_cv_02   |    0.6883    |   0.6883   |     0.7973     |       0.6619      |     0.6194    |    0.6194   |      0.6652     |       0.6538       |
|   test_03_cv_00   |    0.7855    |   0.7855   |     0.7662     |       0.8413      |     0.6413    |    0.6413   |      0.6032     |       0.7821       |
|   test_03_cv_01   |    0.6199    |   0.6199   |     0.5635     |       0.8123      |     0.6731    |    0.6731   |      0.6997     |       0.7158       |
|   test_03_cv_02   |    0.6783    |   0.6783   |     0.8009     |       0.6438      |     0.6169    |    0.6169   |      0.8428     |       0.5326       |
|      **mean**     |    0.6931    |   0.6931   |     0.7316     |       0.7289      |     0.6342    |    0.6342   |      0.6490     |       0.7176       |
|      **std**      |    0.0447    |   0.0447   |     0.0846     |       0.0606      |     0.0301    |    0.0300   |      0.0762     |       0.0667       |
|      **max**      |    0.7855    |   0.7855   |     0.8318     |       0.8413      |     0.6796    |    0.6790   |      0.8428     |       0.7847       |
|      **min**      |    0.6199    |   0.6199   |     0.5635     |       0.6438      |     0.5638    |    0.5638   |      0.5132     |       0.5326       |


### **Scanner differences:**
Table 05. Architecture used: UNet. Patch size: 128x128 pixels (best fold is reported in bold font). Scanner: Hamamatsu NanoZoomer 2.0-RS.
|   **fold_name**   | **dev_dice** | **dev_f1** | **dev_recall** | **dev_precision** |
|:-----------------:|:------------:|:----------:|:--------------:|:-----------------:|
|   test_00_cv_00   |    0.7361    |   0.7354   |      0.693     |       0.8106      |
|   test_00_cv_01   |    0.7234    |   0.7284   |     0.7451     |       0.734       |
| **test_00_cv_02** |  **0.7388**  | **0.7338** |   **0.7155**   |     **0.7744**    |
|   test_00_cv_03   |    0.7384    |    0.732   |     0.7063     |       0.7818      |
|      **mean**     |    0.7342    |   0.7324   |     0.7150     |       0.7752      |
|      **std**      |    0.0063    |   0.0026   |     0.0191     |       0.0274      |
|      **max**      |    0.7388    |   0.7354   |     0.7451     |       0.8106      |
|      **min**      |    0.7234    |   0.7284   |      0.693     |       0.734       |

Table 06. Architecture used: UNet. Patch size: 128x128 pixels (best fold is reported in bold font). Scanner: Hamamatsu NanoZoomer S60.
|   **fold_name**   | **dev_dice** | **dev_f1** | **dev_recall** | **dev_precision** |
|:-----------------:|:------------:|:----------:|:--------------:|:-----------------:|
|   test_00_cv_00   |    0.6286    |   0.6429   |     0.6121     |       0.7143      |
| **test_00_cv_01** |  **0.6757**  | **0.6695** |   **0.6931**   |     **0.6855**    |
|   test_00_cv_02   |     0.617    |   0.6448   |     0.6668     |       0.6443      |
|   test_00_cv_03   |    0.6167    |   0.6445   |     0.6653     |       0.6452      |
|      **mean**     |    0.6345    |   0.6504   |     0.6593     |       0.6723      |
|      **std**      |    0.0243    |   0.0110   |     0.0294     |       0.0294      |
|      **max**      |    0.6757    |   0.6695   |     0.6931     |       0.7143      |
|      **min**      |    0.6167    |   0.6429   |     0.6121     |       0.6443      |


