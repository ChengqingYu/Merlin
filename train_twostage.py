import copy
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import random
import time
from metrics.mask_metric import masked_mae,masked_mape,masked_rmse
from models.forecasting.iTransformer.ITransformer_arch import ITransformer
from models.imputation.TimesNet.TimesNet_arch import TimesNet
from datasets.data_solve import batch_data_solve_one_mask, batch_data_solve_one

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

seed = 3407
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# PEMS04, METR-LA, Global-Wind, China-AQI
data_name = "PEMS04"
model_impu_name = "TimesNet"
model_fore_name = "ITransformer"

### Hyperparameter
input_len = 12
out_len = 12
num_id = 307

timesnet_hypara = {
    "seq_len": input_len,
    "label_len": input_len // 2,  # start token length used in decoder
    "pred_len": out_len,  # prediction sequence length
    "enc_in": num_id,                        # num nodes
    "c_out": num_id,
    "top_k": 5,                                # attn factor
    "d_model": 64,
    "d_ff": 128,
    "num_kernels": 6,
    "e_layers": 2,                              # num of encoder layers
    "embed": "timeF",  # [timeF, fixed, learned]
    "dropout": 0.05,
    "num_time_features": 2,                     # number of used time features
    "time_of_day_size": 288,
    "day_of_week_size": 7,
    "day_of_month_size":None,
    "day_of_year_size": None
}

itrans_hypara = {
    "enc_in": input_len,                        # num nodes
    "dec_in": num_id,
    "c_out": num_id,
    "seq_len": input_len,
    "label_len": input_len//2,       # start token length used in decoder
    "pred_len": out_len,         # prediction sequence length
    'back_size': [9,12,12],
    "vit_layer": 2,
    "factor": 3, # attn factor
    "p_hidden_dims": [128, 128],
    "p_hidden_layers": 2,
    "d_model": 512,                          # window size of moving average. This is a CRUCIAL hyper-parameter.
    "n_heads": 8,
    "e_layers": 4,                              # num of encoder layers
    "d_layers": 1,                              # num of decoder layers
    "d_ff": 512,
    "distil": True,
    "sigma" : 0.2,
    "dropout": 0.1,
    "freq": 'h',
    "use_norm" : True,
    "output_attention": False,
    "embed": "timeF",                           # [timeF, fixed, learned]
    "activation": "gelu",
    }


# Training parameters
miss_rate = 0.5
batch_size = 16
epoch = 101
lr_rate = 0.0002
weight_decay = 0.0001
max_norm = 5
milestone = [1,10,25,50,75,90,100,125, 150, 190]
gamme = 0.5

### Model and Optimizer

imputation_model = TimesNet(timesnet_hypara)
optimizer_impu = optim.Adam(params = imputation_model.parameters(),lr=lr_rate,weight_decay=weight_decay)

forecasting_model = ITransformer(itrans_hypara)
optimizer_fore = optim.Adam(params = forecasting_model.parameters(),lr=lr_rate,weight_decay=weight_decay)

###CPU and GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cpu")

# Load the data
file_line = "datasets/" + data_name + "/data" + str(input_len) + ".npz"
raw_data = np.load(file_line, allow_pickle=True)

### batch
number_train = [i for i in range(raw_data["train_x"].shape[0])]
train_data = DataLoader(number_train,batch_size=batch_size,shuffle=True)

number_vaild = [i for i in range(raw_data["vail_x"].shape[0])]
vaild_data = DataLoader(number_vaild,batch_size=batch_size,shuffle=False)

number_test = [i for i in range(raw_data["test_x"].shape[0])]
test_data = DataLoader(number_test,batch_size=batch_size,shuffle=False)


""""""

print("-----------------------------Training starts------------------------------")

### train, vaild and test data 
feature_train = raw_data["train_x"].astype(np.float64)
feature_vaild = raw_data["vail_x"].astype(np.float64)
feature_test = raw_data["test_x"].astype(np.float64)

target_train = raw_data["train_y"].astype(np.float64)
target_vaild = raw_data["vail_y"].astype(np.float64)
target_test = raw_data["test_y"].astype(np.float64)

if miss_rate == 0.25:
    mask_matric_train = raw_data["train_matric_25"]
    mask_matric_vaild = raw_data["vaild_matric_25"]
    mask_matric_test = raw_data["test_matric_25"]

    print("Missing rate is 25%")

elif miss_rate == 0.5:
    mask_matric_train = raw_data["train_matric_50"]
    mask_matric_vaild = raw_data["vaild_matric_50"]
    mask_matric_test = raw_data["test_matric_50"]

    print("Missing rate is 50%")

elif miss_rate == 0.75:
    mask_matric_train = raw_data["train_matric_75"]
    mask_matric_vaild = raw_data["vaild_matric_75"]
    mask_matric_test = raw_data["test_matric_75"]

    print("Missing rate is 75%")

else:
    mask_matric_train = raw_data["train_matric_90"]
    mask_matric_vaild = raw_data["vaild_matric_90"]
    mask_matric_test = raw_data["test_matric_90"]

    print("Missing rate is 90%")

""""""

print("-----------------------------Training imputation model------------------------------")

imputation_model = imputation_model.to(device)
num_vail = 0
min_vaild_loss = float("inf")

for i in range(epoch):
    imputation_model.train()
    num = 0
    loss_out = 0.0
    start = time.time()

    for data in train_data:

        train_x, train_y, train_x_mask = batch_data_solve_one(feature_train, target_train, mask_matric_train, data.tolist(), device)

        train_impu_x  = imputation_model(train_x_mask, train_y)
        loss_data = masked_mae(train_impu_x, train_x[:,:,:,0], 0.0)

        # Backpropagation and gradient clipping.
        num += 1
        loss_data.backward()

        if max_norm > 0:
            nn.utils.clip_grad_norm_(imputation_model.parameters(), max_norm = max_norm)
        else:
            pass
        optimizer_impu.step()
        loss_out += loss_data

    loss_out = loss_out / num
    end = time.time()

    # Validation set loss
    num_va = 0
    loss_vaild = 0.0
    imputation_model.eval()
    with torch.no_grad():
        for data in vaild_data:

            # vaild_x, vaild_y, vaild_x_mask = batch_data_solve_one_mask(raw_data, data.tolist(), data_name="vaild", mask_rate=miss_rate, device=device)

            vaild_x, vaild_y, vaild_x_mask = batch_data_solve_one(feature_vaild, target_vaild, mask_matric_vaild, data.tolist(), device)

            valid_impu_x = imputation_model(vaild_x_mask, vaild_y)
            loss_data = masked_mae(valid_impu_x, vaild_x[:,:,:,0], 0.0)
            num_va += 1
            loss_vaild += loss_data
        loss_vaild = loss_vaild / num_va

    # Save the weights.
    if loss_vaild < min_vaild_loss:
        min_vaild_loss = loss_vaild
        torch.save(imputation_model.state_dict(), "model_results/" + data_name + "/" + model_impu_name + str(input_len) + ".pth")
    else:
        pass

    # Adjust the learning rate.
    if (i + 1) in milestone:
        for params in optimizer_impu.param_groups:
            params['lr'] *= gamme
    else:
        pass

    print('The {}th epoch, training Loss: {:02.4f}, validation Loss:{:02.4f}, training time:{:02.4f}'.format(i + 1, loss_out, loss_vaild,end - start))



imputation_model.load_state_dict(torch.load("model_results/" + data_name + "/" + model_impu_name + str(input_len) + ".pth"))
imputation_model = imputation_model.to(device).eval()


print("-----------------------------Training forecasting model------------------------------")
forecasting_model = forecasting_model.to(device)
num_vail = 0
min_vaild_loss = float("inf")

for i in range(epoch):
    forecasting_model.train()
    num = 0
    loss_out = 0.0
    start = time.time()

    for data in train_data:

        # train_x, train_y, train_x_mask = batch_data_solve_one_mask(raw_data, data.tolist(), data_name="train", mask_rate=miss_rate, device=device)

        train_x, train_y, train_x_mask = batch_data_solve_one(feature_train, target_train, mask_matric_train, data.tolist(), device)

        with torch.no_grad():
            train_impu_x = imputation_model(train_x_mask, train_y)
            train_impu_x = torch.cat([train_impu_x.unsqueeze(-1), train_x[:,:,:,1:]], dim=-1)

        train_pre  = forecasting_model(train_impu_x, train_y)
        loss_data = masked_mae(train_pre, train_y[:,:,:,0], 0.0)

        # Backpropagation and gradient clipping.
        num += 1
        loss_data.backward()

        if max_norm > 0:
            nn.utils.clip_grad_norm_(forecasting_model.parameters(), max_norm = max_norm)
        else:
            pass
        optimizer_fore.step()
        loss_out += loss_data

    loss_out = loss_out / num
    end = time.time()

    # Validation set loss
    num_va = 0
    loss_vaild = 0.0
    forecasting_model.eval()
    with torch.no_grad():
        for data in vaild_data:

            # vaild_x, vaild_y, vaild_x_mask = batch_data_solve_one_mask(raw_data, data.tolist(), data_name="vaild", mask_rate=miss_rate, device=device)

            vaild_x, vaild_y, vaild_x_mask = batch_data_solve_one(feature_vaild, target_vaild, mask_matric_vaild, data.tolist(), device)

            vaild_impu_x = imputation_model(vaild_x_mask, vaild_y)
            vaild_impu_x = torch.cat([vaild_impu_x.unsqueeze(-1), vaild_x[:,:,:,1:]], dim=-1)

            valid_pre = forecasting_model(vaild_impu_x, vaild_y)
            loss_data = masked_mae(valid_pre, vaild_y[:,:,:,0], 0.0)
            num_va += 1
            loss_vaild += loss_data
        loss_vaild = loss_vaild / num_va

    # Save the weights.
    if loss_vaild < min_vaild_loss:
        min_vaild_loss = loss_vaild
        torch.save(forecasting_model.state_dict(), "model_results/" + data_name + "/" + model_fore_name + str(input_len) + ".pth")
    else:
        pass

    # Adjust the learning rate.
    if (i + 1) in milestone:
        for params in optimizer_fore.param_groups:
            params['lr'] *= gamme
    else:
        pass

    print('The {}th epoch, training Loss: {:02.4f}, validation Loss:{:02.4f}, training time:{:02.4f}'.format(i + 1, loss_out, loss_vaild,end - start))


print('---------------------------------Training completed-------------------------------')


forecasting_model.load_state_dict(torch.load("model_results/" + data_name + "/" + model_fore_name + str(input_len) + ".pth"))
forecasting_model = forecasting_model.to(device)
forecasting_model.eval()

with torch.no_grad():

    all_pre = 0.0
    all_true = 0.0
    num = 0
    for data in test_data:
        
        # test_x, test_y, test_x_mask = batch_data_solve_one_mask(raw_data, data.tolist(), data_name="test", mask_rate=miss_rate, device=device)

        test_x, test_y, test_x_mask = batch_data_solve_one(feature_test, target_test, mask_matric_test, data.tolist(), device)

        test_impu_x = imputation_model(test_x_mask, test_y)
        test_impu_x = torch.cat([test_impu_x.unsqueeze(-1), test_x[:,:,:,1:]], dim=-1)
        test_pre  = forecasting_model(test_impu_x, test_y)

        if num == 0:
            all_pre = test_pre.to(device2)
            all_true = test_y[:,:,:,0].to(device2)
        else:
            all_pre = torch.cat([all_pre, test_pre.to(device2)], dim=0)
            all_true = torch.cat([all_true, test_y[:,:,:,0].to(device2)], dim=0)
        num += 1

# denormalization
def Inverse_normalization(x,max,min):
    return x * (max - min) + min

final_pred = Inverse_normalization(all_pre, raw_data["max_min"][0],raw_data["max_min"][1])
final_target = Inverse_normalization(all_true, raw_data["max_min"][0],raw_data["max_min"][1])

mae,mape,rmse = masked_mae(final_pred, final_target,0.0), masked_mape(final_pred, final_target,0.0)*100,masked_rmse(final_pred, final_target,0.0)
print('The metrics of two stage model ({} + {}) when missing rate is {}%: \nRMSE: {:02.4f}, MAPE: {:02.4f}, MAE: {:02.4f}'.format(model_impu_name, model_fore_name, miss_rate * 100, rmse, mape, mae))