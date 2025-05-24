# Merlin
Code for our SIGKDD'25 paper "Merlin: Multi-View Representation Learning for Robust Multivariate Time Series Forecasting with Unfixed Missing Rates"

If the code is helpful to you, please cite the following paper:
```bibtex
@inproceedings{yu2025merlin,
  title={Merlin: Multi-View Representation Learning for Robust Multivariate Time Series Forecasting with Unfixed Missing Rates},
  author={Yu, Chengqing and Wang, Fei and Yang, Chuanguang and Shao, Zezhi and Sun, Tao and Qian, Tangwen and Wei, Wei and An, Zhulin and Xu, Yongjun},
  booktitle = {SIGKDD},
  year={2025}
}
```

## Requirements
This code is built on Python 3.11. The required packages can be installed using the following command:
```bash
# Install Python
conda create -n Merlin python=3.11
conda activate Merlin
# Install PyTorch
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# Install other dependencies
pip install -r requirements.txt
```

## Data processing
The four datasets (METR-LA, PEMS04, China AQI, and Global Wind) adopted in our paper can be found at the following two links:

Our repository provides compressed files for part of the data. Please extract them before running the data preprocessing code.

This paper presents two data masking ways:
1. The masked points in the low missing rate dataset are a subset of those in the high missing rate dataset.
2. Both the low and high missing rate datasets are masked randomly.

This project places the first masking approach in the `datasets/` directory. If you wish to use the second masking approach, please move the `datasets/PEMS04/preprocessing_PEMS04_random.py` file from the data folder to the `datasets/` directory before running.

Please use the following way to process the raw data:
```bash
# METR-LA dataset
python datasets/preprocessing_METR_subset.py
# PEMS04 dataset
python datasets/preprocessing_PEMS04_subset.py
# China AQI dataset
python datasets/preprocessing_AQI_subset.py
# Global Wind dataset
python datasets/preprocessing_WIND_subset.py
```
## The hyperparameters of our model
### STID
1. input_length: 12 (This paper follows the experimental settings of most existing spatiotemporal prediction models, fixing both the length of historical observations and future predictions to 12.)
2. out_length: 12
3. num_nodes: number of time series ()
4. embedding_size: 64
5. node_embedding: 64
6. input_size: 3

### Merlin
1. Weight_pre: 2 (The weight of the L1 loss.)
2. weight_KD: 2 (The weight of knowledge distillation.)
3. weight_CL: 1 (The weight of contrastive loss.)
4. Temperature: 1
5. number_missing_rates: 4 (We use four missing rates in this paper)
6. batch size: 16
7. epoch: 101

## Train the teacher model and the student model
After completing data preprocessing, please train the teacher model using the following way:
```bash
python train_teacher.py
```
This repository provides pretrained teacher model weight files for four datasets. For details, please refer to directories such as `model_results/PEMS04`.

After training the teacher model, please train the student model using the following way:
```bash
python train_student_Merlin.py
```
After running the above file, the model's performance metrics under the four missing rates will be printed directly.

## Train one stage model
If you want to train a separate forecasting model for a specific missing rate, you can run and debug the following code：
```bash
python train_onestage.py
```

## Train two stage model
If you want to train a two stage model (imputation + forecasting) for a specific missing rate, you can run and debug the following code：
```bash
python train_twostage.py
```

## Folder Structure:

The folder structure is organized as follows::

```
Merlin/
├── datasets/                  # Contains processed data and masking scripts
│   ├── METR-LA/
│   ├── PEMS04/
│   ├── China-AQI/
│   ├── Global-wind/
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
│   │   └── Itransformer/
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
└── README.md                 # Project overview and instructions
```

## Acknowledge
We appreciate the following github repos for their valuable codebase:
- STID: https://github.com/thuml/Time-Series-Library
- iTransformer and TimesNet: https://github.com/GestaltCogTeam/BasicTS
- TSmixer: https://github.com/google-research/google-research/tree/master/tsmixer





