# Merlin
Code for our SIGKDD'25 paper "Merlin: Multi-View Representation Learning for Robust Multivariate Time Series Forecasting with Unfixed Missing Rates"

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

This paper presents two data masking ways:
1. The masked points in the low missing rate dataset are a subset of those in the high missing rate dataset.
2. Both the low and high missing rate datasets are masked randomly.
This project places the first masking approach in the `datasets/` directory. If you wish to use the second masking approach, please move the `preprocessing_METR_random.py` file from the data folder to the `datasets/` directory before running.
Please process the data using the following way:
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


