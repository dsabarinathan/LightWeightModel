
# Light Weight Residual Dense Attention Net for Spectral Reconstruction

This is implementation of LWRDA Net
["Light Weight Residual Dense Attention Net for Spectral Reconstruction from
RGB Images by K.Uma et .2020"](https://arxiv.org/ftp/arxiv/papers/2004/2004.06930.pdf) 

## Environment

1. Python 3.6.1
2. Anaconda 5.0.1
3. Ubuntu 16.04 or Windows10

## How to setup the environment

#### Step 1 

Unzip the downloaded folder


#### Step 2

Open the powershell or terminal


#### Step 3

```
$cd yourpathtoLightWeightModel

$pwd
> ~/LightWeightModel

$pip install --upgrade -r requirements.txt

```
## How to test the model on your own imgaes
```
$python test.py --testImagePath=yourpathtoimages
```

## Results

| Data size  | Data  |  MRAE  |  SSIM  |
| :------: | :------: | :-------: | :-------: |  
| 400  | Training Data  | 0.02372  | 0.9899  |
| 50  | Validation Data  | 0.04497  | 0.9827  |
| 10  | Testing Data1  | 0.05478 | - |
| 10  | Testing Data2 | 0.04577  | -  |



