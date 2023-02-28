# Pytorch Adain Style
 PyTorch Implementation of the paper "[Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)"

***Notice: This repositories is simple implementation for self-learning***

## Environment

- Python >= 3.8
- Pytorch >= 1.8

``` bash
pip install -r requirements.txt
```

## Dataset

- Content images: [COCO](https://cocodataset.org/#download)
- Style images: [Wikiart](https://www.kaggle.com/c/painter-by-numbers)

### Data structure

``` text
-project_name/
    -contents/
        -*.png(or others format)
    -styles/
        -*.png(or others format)
        
# for example
# The number of content images must be consistent with the number of style images
-Adain/
    -contents/
        -00001.png
        -00002.png
        -00003.png
    -styles/
        -00004.png
        -00005.png
        -00006.png
```

## Training

``` python
python train.py --train-content contents --train-style styles --gpu 0,1 --workers 8 --epochs 100 --batch-size 16
```

## Test

``` python
# source is directory
python test.py --weights epoch_100.pth --content contents_path --style styles_path

# source is file
python test.py --weights epoch_100.pth --content 0000.1.png --style 0000.2.png
```

## Reference

- [Original implementation](https://github.com/xunhuang1995/AdaIN-style)
- [irasin's implementation](https://github.com/irasin/Pytorch_AdaIN)
