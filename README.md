<div align="center">
<h2>A novel non-pretrained deep supervision network for polyp segmentation</h2>
Zhenni Yu, Li Zhao, Tangfei Liao, Xiaoqin Zhang, Geng Chen, Guobao Xiao
Pattern Recognition (PR), 2024, 154, 110554
</div>

## Usage 

### Installation

```bash
git clone https://github.com/guobaoxiao/NPD-Net
cd NPD-Net
```
## Pre-trained models
- **Traindata**:
    download the traindata set from [here](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view?usp=sharing), put into 'data/TrainDataset'
- **Testdata**:
    download the traindata set from [here](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view?usp=sharing), put into 'data/TestDataset'
- **pre-weigth**:
    download the weight of res2net50 from [here](https://drive.google.com/file/d/1_1N-cx1UpRQo7Ybsjno1PAg4KE1T9e5J/view?usp=sharing), put into 'weigth/'
    download the weight of pvtv2 from [here](https://github.com/whai362/PVT/releases/tag/v2), put into 'weigth/'


### Weights
- **pre-weigth**:
    gt_encoder_model: loss-10.5362-GTENCODER-NPD_Net-1.pth, put into 'model_path'

    链接：https://pan.baidu.com/s/1p5UBnN5lJFKABD-mmp0xnw 
    提取码：cevt

### Train
```bash
python Mytrain.py
```

### Test

```bash
python Mytest.py
```

- **For miou and mDice**:
```bash
python eval.py
```

- **For more detailed metrics** :
```bash
python MSCAF_COD_evaluation/evaluation.py
```
## Citation

If you find this project useful, please consider citing:

```bibtex
@article{yu2024novel,
  title={A novel non-pretrained deep supervision network for polyp segmentation},
  author={Yu, Zhenni and Zhao, Li and Liao, Tangfei and Zhang, Xiaoqin and Chen, Geng and Xiao, Guobao},
  journal={Pattern Recognition},
  pages={110554},
  year={2024},
  publisher={Elsevier}
}
```
