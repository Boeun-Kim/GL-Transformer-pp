# GL-Transformer++ for Group Activity Recognition (Pattern Recognition 2025)
This is an official implementation of "A Unified Framework for Unsupervised Action Learning via Global-to-Local Motion Transformer" [[paper]](https://www.sciencedirect.com/science/article/pii/S0031320324008690). GL-Transformer++ is an extension method of GL-Transformer (ECCV 2022) [[paper]](https://arxiv.org/abs/2207.06101) [[project]](https://boeun-kim.github.io/).




![architecture](https://github.com/Boeun-Kim/GL-Transformer-pp/blob/main/figures/architecture.png)



 ## Dependencies

We tested our code on the following environment.

- CUDA 11.3
- python 3.8.10
- pytorch 1.11.0

Install python libraries with:

```
pip install -r requirements.txt
```




## Data preparation

1. Download dataset from https://drive.google.com/file/d/1iey8R5ZgDLGMqWdJ9vJBb3VH6dT5joMY/view into `./data`

   - volleyball.zip

3. Unzip the data

   ```
   cd ./data
   unzip volleyball.zip
   ```
   
4. Preprocess the data

   ```
   python volley_gendata.py
   ```




## Unsupervised Pretraining

Sample arguments for unsupervised pretraining:

(please refer to `arguments.py` for detailed arguments.)

```
python learn_PTmodel.py \
    --data_path ./data/preprocessed \
    --save_path [pretrained weights saving path]
```




## Linear Evaluation Protocol

Sample arguments for training and evaluating a linear classifier:

(please refer to `arguments.py` for detailed arguments.)

```
python linear_eval_protocol.py \
    --data_path ./data/preprocessed \
    --pretrained_model [pretrained weights path] \
    --save_path [whole model weights saving path]
```




## Download trained weights

Trained weights can be downloaded via

GL-Transformer++ Large

https://drive.google.com/file/d/15Ahsq5zroIBRV4JWpb7OgipNh8BW1BZb/view?usp=drive_link

GL-Transformer++ Small

https://drive.google.com/file/d/1RWRo-ilp3rtdM4qVOXv2dxvJhgrLfBxy/view?usp=drive_link

| Model                  | GFLOPs | Accuracy (%) |
| ---------------------- | ------ | ------------ |
| GL-Transformer++ Small | 0.55   | 80.1         |
| GL-Transformer++ Large | 20.00  | 88.0         |






## Test for Action Recognition

Sample arguments for testing whole framework:

(please refer to `arguments.py` for detailed arguments.)

```
python test_actionrecog.py \
    --data_path ./data/preprocessed \
    --pretrained_model_w_classifier pretrained/linear/PT_w_classifier
```

To test GL-Transformer++ Small, arguments should be

```
python test_actionrecog.py \
    --data_path ./data/preprocessed \
    --num_heads 2 --dim_emb 6 --ff_expand 1 \
    --pretrained_model_w_classifier pretrained_small/linear/PT_w_classifier
```




## Reference

Part of our code is based on [COMPOSER](https://github.com/hongluzhou/composer).

Thanks to the great resources.



## Citation

Please cite our work if you find it useful.

```
@inproceedings{kim2022global,
  title={Global-local motion transformer for unsupervised skeleton-based action learning},
  author={Kim, Boeun and Chang, Hyung Jin and Kim, Jungho and Choi, Jin Young},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part IV},
  pages={209--225},
  year={2022},
  organization={Springer}
}
```

```
@article{kim2024unified,
  title={A unified framework for unsupervised action learning via global-to-local motion transformer},
  author={Kim, Boeun and Kim, Jungho and Chang, Hyung Jin and Oh, Tae-Hyun},
  journal={Pattern Recognition},
  pages={111118},
  year={2024},
  publisher={Elsevier}
}
```
