


VLN-Trans (https://arxiv.org/pdf/2302.09230.pdf)

  
  

### Installation
Install the [Matterport3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator). 


 
  

### Data Preparation

Please follow the instructions below to prepare the data in directories:

  
- MP3D navigability graphs:  [connectivity maps](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/connectivity).

  
- Processed fine-grained R2R data and augmented: [Fine-grained r2r data](https://drive.google.com/file/d/1RNyHIxbG67v28ll1iaZbfrIunfkjO0ps/view?usp=share_link).

 
- Processed fine-grained R4R data and augmented: [Fine-grained r4r data](https://drive.google.com/file/d/1RNyHIxbG67v28ll1iaZbfrIunfkjO0ps/view?usp=share_link).

  

- R2R-Last: [R2R-Last](https://drive.google.com/file/d/1MmDWz0JG0DlF5qr601kTT25wwcCDPon5/view?usp=share_link).

  

- Translator pre-train data: [Translator pre-train data](https://drive.google.com/file/d/1RNyHIxbG67v28ll1iaZbfrIunfkjO0ps/view?usp=share_link).

  

- MP3D image features: [img features](https://www.dropbox.com/s/85tpa6tc3enl5ud/ResNet-152-places365.zip?dl=1) (ResNet-152-Places365).

  

### Initial weights for Rec-VLN

 
- Pre-trained weights

- Download the `base-no-labels` following [this guide](https://github.com/microsoft/Oscar/blob/master/DOWNLOAD.md).

- Pre-trained [PREVALENT](https://github.com/weituo12321/PREVALENT) weights

- Download the `pytorch_model.bin` from [here](https://drive.google.com/drive/folders/1sW2xVaSaciZiQ7ViKzm_KbrLD_XvOq5y).

  

### Trained Network Weights

  

- VLN-trans [trained-weights](https://drive.google.com/file/d/1W7hDGTvKXeXKX-gtyKujTcYf7hpARG_b/view?usp=share_link)

- VLN-trans pre-train weights [pre-train initial weights for the agent](https://drive.google.com/file/d/1x0szprQKmyts9PvdvunS-trYJtEb9Qt9/view?usp=share_link)

- Translator pre-train weights [pre-train weights for the translator](https://drive.google.com/file/d/1ZF9yFh6axZiRCORT4vQktxTlRb2NUjBE/view?usp=share_link)

  

## R2R Navigation

  

Please read Peter Anderson's VLN paper for the [R2R Navigation task](https://arxiv.org/abs/1711.07280).

  

### Test Navigator

  

To replicate the performance reported in our paper, load the trained network weights and run validation:

```bash

bash run_translator/test_agent.bash

```


  

### Train Navigator

  

To train the network from scratch, simply run:

```bash

bash run_translator/train_agent.bash

```

The trained Navigator will be saved under `snap/`.
