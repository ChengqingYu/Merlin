# Merlin
Code for our SIGKDD'25 paper "Merlin: Multi-View Representation Learning for Robust Multivariate Time Series Forecasting with Unfixed Missing Rates"

## Requirements
This code is built on Python 3.11. The required packages can be installed using the following command:
```bash
# Install Python
conda create -n Merlin python=3.11
conda activate Merlin
# Install dependencies
pip install -r requirements.txt
```

## Data processing
The four datasets (METR-LA, PEMS04, China AQI, and Global Wind) adopted in our paper can be found at the following two links:

Our repository provides compressed files for part of the data. Please extract them before running the data preprocessing code.

This paper presents two data masking ways:
1. The masked points in the low missing rate dataset are a subset of those in the high missing rate dataset.
2. Both the low and high missing rate datasets are masked randomly.

This project places the first masking approach in the `datasets/` directory. If you wish to use the second masking approach, please move the `datasets/PEMS04/preprocess_PEMS04_radom.py` file from the data folder to the `datasets/` directory before running. The processing way for other datasets is the same.

Please use the following way to process the raw data:
```bash
# METR-LA dataset
python datasets/preprocess_METR_subset.py
# PEMS04 dataset
python datasets/preprocess_PEMS04_subset.py
# China AQI dataset
python datasets/preprocess_AQI_subset.py
# Global Wind dataset
python datasets/preprocess_WIND_subset.py
```
## The hyperparameters of our model
### STID
1. input_len: 12 (This paper follows the experimental settings of most existing spatiotemporal forecasting models, fixing both the length of historical observations and future forecasting to 12.)
2. output_len: 12
3. num_nodes: number of time series
4. input_size: 3 (It represents the dimensions after concatenating the original time series with the two temporal embeddings, set to 3.)
5. if_T_i_D: True
6. if_D_i_W: True
7. embed_dim: 64 (For the basic introduction to these spatiotemporal embeddings, please refer to the original STID paper: https://github.com/GestaltCogTeam/STID.)
8. node_dim: 64
9. temp_dim_tid: 64
10. temp_dim_diw: 64
11. time_of_day_size: METR-LA and PEMS04: 288, China AQI: 24, Global wind: 1
12. day_of_week_size: 7
13. cl_hidden = 4  (Since contrastive learning requires dimensionality reduction of the representations, we provide two methods. The representation has the dimension [B, H, N, 1]. The first method reduces it to [B, H2, N], while the second method reduces it to [B, H2, 1].)
14. cl_student = 16

### Merlin
1. Weight_pre: 2 (The weight of the L1 loss.)
2. weight_KD: 2 (The weight of knowledge distillation.)
3. weight_CL: 1 (The weight of contrastive loss.)
4. Temperature: 1
5. num_miss_rate: 4 (We use four missing rates in this paper)
6. batch size: 16
7. epoch: 101
8. max_norm = 5 (Gradient clipping.)
9. lr_rate = 0.0002
10. weight_decay = 0.0001
11. milestone = [1,10,25,50,75,90,100] (Adjust the learning rate when the epoch reaches the numbers in the list.)
12. gamme = 0.5 (Learning rate adjustment ratio.)

## Train the teacher model and the student model
After completing data preprocessing, please train the teacher model using the following way:
```bash
python train_teacher.py
```
This repository provides pretrained teacher model weight files for four datasets. For details, please refer to directories such as `model_results/PEMS04`. Due to the large size of the global wind files, we have not uploaded them to the repository yet.

After training the teacher model, please train the student model using the following way:
```bash
python train_student_merlin.py
```
After running the above file, the model's performance metrics under the four missing rates will be printed directly.

## Train one stage model
We provid TSmixer as an example.
If you want to train a separate forecasting model for a specific missing rate, you can run and debug the following code：
```bash
python train_onestage.py
```

## Train two stage model
We provid TimesNet as the imputation and iTransformer as forecasting.
If you want to train a two stage model (imputation + forecasting) for a specific missing rate, you can run and debug the following code：
```bash
python train_twostage.py
```
## citation
If the code is helpful to you, please cite the following paper:
```bibtex
@inproceedings{yu2025merlin,
  title={Merlin: Multi-View Representation Learning for Robust Multivariate Time Series Forecasting with Unfixed Missing Rates},
  author={Yu, Chengqing and Wang, Fei and Yang, Chuanguang and Shao, Zezhi and Sun, Tao and Qian, Tangwen and Wei, Wei and An, Zhulin and Xu, Yongjun},
  booktitle = {SIGKDD},
  year={2025}
}
```

## Folder Structure:

The folder structure is organized as follows::

```
Merlin/
├── datasets/                  # Contains processed data and masking scripts
│   ├── METR-LA/
│   │   ├── METR-LA.h5
│   │   └── METR-LA.pkl
│   ├── PEMS04/
│   │   ├── adj_PEMS04_distance.pkl
│   │   ├── adj_PEMS04.pkl
│   │   ├── PEMS04.csv
│   │   └── PEMS04.npz
│   ├── China-AQI/
│   │   └── China-AQI.csv
│   ├── Global-wind/
│   │   └── Global-Wind.csv
│   ├── preprocessing_METR_subset.py
│   ├── preprocessing_PEMS_subset.py
│   ├── preprocessing_AQI_subset.py
│   ├── preprocessing_WIND_subset.py
│   └── data_solve.py
│
├── model_results/            # Stores trained model weights
│   ├── PEMS04/
│   ├── METR-LA/
│   ├── China-AQI/
│   └── Global-Wind/
│
├── models/                  # Model architecture definitions
│   ├── forecasting/
│   │   ├── STID/
│   │   ├── TSmixer/
│   │   └── iTransformer/
│   └── imputation/
│       └── TimesNet/
│                        
├── Metircs/                # Loss (Merlin) and Metrics   
│   ├── Loss.py
│   └── mask_metrics.py
│
├── python train_teacher.py
├── python train_student_Merlin.py
├── python train_onestage.py
├── python train_twostage.py
├── requirement.txt
└── README.md                 # Project overview and instructions
```

## Acknowledge
We appreciate the following github repos for their valuable codebase:
- STID: https://github.com/GestaltCogTeam/BasicTS
- iTransformer and TimesNet: https://github.com/thuml/Time-Series-Library
- TSmixer: https://github.com/google-research/google-research/tree/master/tsmixer





