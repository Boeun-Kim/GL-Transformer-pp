# GL-Transformer++ (under review)

This is the official implementation of "A Unified Framework for Unsupervised Action Learning via Global-to-Local Motion Transformer". This is an extension method of GL-Transformer [[paper]](https://arxiv.org/abs/2207.06101) [[project]](https://boeun-kim.github.io/).




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

https://drive.google.com/file/d/15Ahsq5zroIBRV4JWpb7OgipNh8BW1BZb/view?usp=drive_link




## Test for Action Recognition

Sample arguments for testing whole framework:

(please refer to `arguments.py` for detailed arguments.)

```
python test_actionrecog.py \
    --data_path ./data/preprocessed \
    --pretrained_model_w_classifier pretrained/linear/PT_w_classifier
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
