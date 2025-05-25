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
from models.forecasting.TSmixer.TSMixer_arch import TSMixer
from datasets.data_solve import batch_data_solve_one_mask, batch_data_solve_one

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

seed = 3407
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

### Hyperparameter
# PEMS04, METR-LA, Global-Wind, China-AQI
data_name = "PEMS04"
model_name = "TSMixer"

# Model parameters
input_len= 12
pred_len = 12
ff_dim = 2048
vare_num = 307
n_block = 2
dropout = 0.05


# Training parameters
miss_rate = 0.25
batch_size = 32
epoch = 101
lr_rate = 0.002
weight_decay = 0.0001
max_norm = 5
milestone = [1,10,25,50,75,90,100]
gamme = 0.5

### Model and Optimizer
my_net = TSMixer(input_len, vare_num, pred_len, n_block, dropout, ff_dim)
optimizer = optim.Adam(params=my_net.parameters(),lr=lr_rate,weight_decay=weight_decay)

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

my_net = my_net.to(device)
num_vail = 0
min_vaild_loss = float("inf")

for i in range(epoch):
    my_net.train()
    num = 0
    loss_out = 0.0
    start = time.time()

    for data in train_data:

        # train_x, train_y, train_x_mask = batch_data_solve_one_mask(raw_data, data.tolist(), data_name="train", mask_rate=miss_rate, device=device)

        train_x, train_y, train_x_mask = batch_data_solve_one(feature_train, target_train, mask_matric_train, data.tolist(), device)

        train_pre  = my_net(train_x_mask)

        loss_data = masked_mae(train_pre, train_y[:,:,:,0], 0.0)

        # Backpropagation and gradient clipping.
        num += 1
        loss_data.backward()

        if max_norm > 0:
            nn.utils.clip_grad_norm_(my_net.parameters(), max_norm = max_norm)
        else:
            pass
        optimizer.step()
        loss_out += loss_data

    loss_out = loss_out / num
    end = time.time()

    # Validation set loss
    num_va = 0
    loss_vaild = 0.0
    my_net.eval()
    with torch.no_grad():
        for data in vaild_data:

            # vaild_x, vaild_y, vaild_x_mask = batch_data_solve_one_mask(raw_data, data.tolist(), data_name="vaild", mask_rate=miss_rate, device=device)

            vaild_x, vaild_y, vaild_x_mask = batch_data_solve_one(feature_vaild, target_vaild, mask_matric_vaild, data.tolist(), device)

            valid_pre = my_net(vaild_x_mask)
            loss_data = masked_mae(valid_pre, vaild_y[:,:,:,0], 0.0)
            num_va += 1
            loss_vaild += loss_data
        loss_vaild = loss_vaild / num_va

    # Save the weights.
    if loss_vaild < min_vaild_loss:
        min_vaild_loss = loss_vaild
        torch.save(my_net.state_dict(), "model_results/" + data_name + "/" + model_name + ".pth")
    else:
        pass

    # Adjust the learning rate.
    if (i + 1) in milestone:
        for params in optimizer.param_groups:
            params['lr'] *= gamme
    else:
        pass

    print('The {}th epoch, training Loss: {:02.4f}, validation Loss:{:02.4f}, training time:{:02.4f}'.format(i + 1, loss_out, loss_vaild,end - start))


print('---------------------------------Training completed-------------------------------')


my_net.load_state_dict(torch.load("model_results/" + data_name + "/" + model_name + ".pth"))
my_net = my_net.to(device)
my_net.eval()

with torch.no_grad():

    all_pre = 0.0
    all_true = 0.0
    num = 0
    for data in test_data:
        
        # test_x, test_y, test_x_mask = batch_data_solve_one_mask(raw_data, data.tolist(), data_name="test", mask_rate=miss_rate, device=device)

        test_x, test_y, test_x_mask = batch_data_solve_one(feature_test, target_test, mask_matric_test, data.tolist(), device)
    
        test_pre  = my_net(test_x_mask)

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
print('The metrics of one stage model ({}) when Missing rate is {}%: \nRMSE: {:02.4f}, MAPE: {:02.4f}, MAE: {:02.4f}'.format(model_name, miss_rate * 100, rmse, mape, mae))