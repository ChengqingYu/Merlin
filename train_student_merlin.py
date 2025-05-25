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
from datasets.data_solve import batch_data_solve_all_mask, batch_data_solve_student
from metrics.Loss import Merlin_Loss_mutilstage, Merlin_Loss_add
from models.forecasting.STID.stid_arch import STID, STID_CL

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

seed = 3407
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


# Load the data
# PEMS04, METR-LA, Global-Wind, China-AQI
data_name = "PEMS04"
model_name_teacher = "STID_Teacher"
model_name_student = "STID_Student_merlin"

# Training parameters
epoch_Thre = 3
num_miss_rate = 4
weight_pre = 2
weight_KD = 2
weight_CL = 1
temperature = 1
batch_size = 16
epoch = 101 + epoch_Thre
max_norm = 5
lr_rate = 0.0002
weight_decay = 0.0001
milestone = [1,10,25,50,75,90,100]
gamme = 0.5

### Hyperparameter
num_nodes=307
input_len= 12
input_dim= 3
embed_dim= 64
cl_hidden = 4
cl_student = 16
output_len= 12
num_layer=3
if_node=True
node_dim= 64
if_T_i_D = True
if_D_i_W = True
temp_dim_tid=64
temp_dim_diw=64
time_of_day_size=288
day_of_week_size=7

### Model and Optimizer
teacher_net = STID(num_nodes,node_dim,input_len, input_dim,embed_dim,
                 output_len,num_layer,cl_hidden,temp_dim_tid,temp_dim_diw,time_of_day_size,
                 day_of_week_size,if_T_i_D,if_D_i_W,if_node)
teacher_net.load_state_dict(torch.load("model_results/" + data_name + "/" + model_name_teacher + str(input_len) + ".pth"))

student_net = STID_CL(num_nodes,node_dim,input_len, input_dim,embed_dim,
                 output_len,num_layer,cl_student,temp_dim_tid,temp_dim_diw,time_of_day_size,
                 day_of_week_size,if_T_i_D,if_D_i_W,if_node)
optimizer = optim.Adam(params=student_net.parameters(),lr=lr_rate,weight_decay=weight_decay)

# CPU and GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cpu")

# Load the data
file_line = "datasets/" + data_name + "/data" + str(input_len) + ".npz"
raw_data = np.load(file_line, allow_pickle=True)

# batch, input and target

number_train = [i for i in range(raw_data["train_x"].shape[0])]
train_data = DataLoader(number_train,batch_size=batch_size,shuffle=True)

number_vaild = [i for i in range(raw_data["vail_x"].shape[0])]
vaild_data = DataLoader(number_vaild,batch_size=batch_size,shuffle=False)

number_test = [i for i in range(raw_data["test_x"].shape[0])]
test_data = DataLoader(number_test,batch_size=batch_size,shuffle=False)

### train, vaild and test data 
feature_train = raw_data["train_x"].astype(np.float64)
feature_vaild = raw_data["vail_x"].astype(np.float64)
feature_test = raw_data["test_x"].astype(np.float64)

target_train = raw_data["train_y"].astype(np.float64)
target_vaild = raw_data["vail_y"].astype(np.float64)
target_test = raw_data["test_y"].astype(np.float64)

### MASK matric
mask_matric_train25 = raw_data["train_matric_25"]
mask_matric_vaild25 = raw_data["vaild_matric_25"]
mask_matric_test25 = raw_data["test_matric_25"]


mask_matric_train50 = raw_data["train_matric_50"]
mask_matric_vaild50 = raw_data["vaild_matric_50"]
mask_matric_test50 = raw_data["test_matric_50"]


mask_matric_train75 = raw_data["train_matric_75"]
mask_matric_vaild75 = raw_data["vaild_matric_75"]
mask_matric_test75 = raw_data["test_matric_75"]


mask_matric_train90 = raw_data["train_matric_90"]
mask_matric_vaild90 = raw_data["vaild_matric_90"]
mask_matric_test90 = raw_data["test_matric_90"]


print("-----------------------------Training starts------------------------------")

teacher_net = teacher_net.to(device).eval()
student_net = student_net.to(device)
num_vail = 0
min_vaild_loss = float("inf")

for i in range(epoch):
    student_net.train()
    num = 0
    loss_out = 0.0
    start = time.time()
    
    # training sets in each epoch
    for data in train_data:

        train_x, train_y, train_x_mask25, train_x_mask50, train_x_mask75, train_x_mask90 = batch_data_solve_student(feature_train, target_train, mask_matric_train25, mask_matric_train50, mask_matric_train75, mask_matric_train90, data.tolist(), device=device)
        
        # teacher model generate forecasting results and representation
        with torch.no_grad():
            train_Teacher_pre, train_Teacher_hidden, _  = teacher_net(train_x, train_y)

        
        # student model generate forecasting results and representation for each missing rates
        train_student_pre25, train_student_hidden25, train_student_CL_hidden25 = student_net(train_x_mask25)
        train_student_pre50, train_student_hidden50, train_student_CL_hidden50 = student_net(train_x_mask50)
        train_student_pre75, train_student_hidden75, train_student_CL_hidden75 = student_net(train_x_mask75)
        train_student_pre90, train_student_hidden90, train_student_CL_hidden90 = student_net(train_x_mask90)

        # Calculate the loss
        loss_data = Merlin_Loss_add(train_Teacher_pre, train_Teacher_hidden, 
                                train_student_pre25, train_student_hidden25, train_student_CL_hidden25,  
                                train_student_pre50, train_student_hidden50, train_student_CL_hidden50,
                                train_student_pre75, train_student_hidden75, train_student_CL_hidden75,
                                train_student_pre90, train_student_hidden90, train_student_CL_hidden90,
                                train_y[:,:,:,0], num_miss_rate, temperature , weight_KD, weight_CL, weight_pre, i+1, epoch_Thre)


        # Backpropagation and gradient clipping.
        num += 1
        loss_data.backward()

        if max_norm > 0:
            nn.utils.clip_grad_norm_(student_net.parameters(), max_norm = max_norm)
        else:
            pass
        optimizer.step()
        loss_out += loss_data

    loss_out = loss_out / num
    end = time.time()

    # Validation set
    num_va = 0
    loss_vaild = 0.0
    student_net.eval()
    with torch.no_grad():
        for data in vaild_data:
            # vaild_x, vaild_y, vaild_x_mask = batch_data_solve_all_mask(raw_data, data.tolist(), data_name="vaild", device=device)

            vaild_x, vaild_y, vaild_x_mask25, vaild_x_mask50, vaild_x_mask75, vaild_x_mask90 = batch_data_solve_student(feature_vaild, target_vaild, mask_matric_vaild25, mask_matric_vaild50, mask_matric_vaild75, mask_matric_vaild90, data.tolist(), device=device)

            valid_pre25, _, _ = student_net(vaild_x_mask25)
            valid_pre50, _, _ = student_net(vaild_x_mask50)
            valid_pre75, _, _ = student_net(vaild_x_mask75)
            valid_pre90, _, _ = student_net(vaild_x_mask90)

            loss_data = 0.25 * (masked_mae(valid_pre25, vaild_y[:,:,:,0], 0.0) + masked_mae(valid_pre50, vaild_y[:,:,:,0], 0.0) + masked_mae(valid_pre75, vaild_y[:,:,:,0], 0.0) + masked_mae(valid_pre90, vaild_y[:,:,:,0], 0.0))

            num_va += 1
            loss_vaild += loss_data
        loss_vaild = loss_vaild / num_va

    # Save the weights.
    if loss_vaild < min_vaild_loss:
        min_vaild_loss = loss_vaild
        torch.save(student_net.state_dict(),"model_results/" + data_name + "/" + model_name_student + str(input_len) + ".pth")
    else:
        pass

    
    # Adjust the learning rate.
    if (i - epoch_Thre + 1) in milestone:
        for params in optimizer.param_groups:
            params['lr'] *= gamme
    else:
        pass

    print('The {} th epoch, training Loss: {:02.4f}, validation Loss:{:02.4f}, training time:{:02.4f}'.format(i + 1, loss_out, loss_vaild,end - start))


print('---------------------------------Training completed-------------------------------')

student_net.load_state_dict(torch.load("model_results/" + data_name + "/" + model_name_student + str(input_len) + ".pth"))
student_net = student_net.to(device)
student_net.eval()

# Forecasting results of test sets
with torch.no_grad():

    all_pre25 = 0.0
    all_pre50 = 0.0
    all_pre75 = 0.0
    all_pre90 = 0.0
    all_true  = 0.0

    num = 0
    for data in test_data:
        
        # test_x, test_y, test_x_mask = batch_data_solve_all_mask(raw_data, data.tolist(), data_name="test", device=device)
        test_x, test_y, test_x_mask25, test_x_mask50, test_x_mask75, test_x_mask90 = batch_data_solve_student(feature_test, target_test, mask_matric_test25, mask_matric_test50, mask_matric_test75, mask_matric_test90, data.tolist(), device=device)
        
        test_pre25, _,_ = student_net(test_x_mask25)
        test_pre50, _,_ = student_net(test_x_mask50)
        test_pre75, _,_ = student_net(test_x_mask75)
        test_pre90, _,_ = student_net(test_x_mask90)

        if num == 0:
            all_pre25 = test_pre25.to(device2)
            all_pre50 = test_pre50.to(device2)
            all_pre75 = test_pre75.to(device2)
            all_pre90 = test_pre90.to(device2)

            all_true = test_y[:,:,:,0].to(device2)
        else:
            all_pre25 = torch.cat([all_pre25, test_pre25.to(device2)], dim=0)
            all_pre50 = torch.cat([all_pre50, test_pre50.to(device2)], dim=0)
            all_pre75 = torch.cat([all_pre75, test_pre75.to(device2)], dim=0)
            all_pre90 = torch.cat([all_pre90, test_pre90.to(device2)], dim=0)
            all_true  = torch.cat([all_true,  test_y[:,:,:,0].to(device2)], dim=0)
        num += 1


# denormalization
def Inverse_normalization(x,max,min):
    return x * (max - min) + min

final_pred25 = Inverse_normalization(all_pre25, raw_data["max_min"][0], raw_data["max_min"][1])
final_pred50 = Inverse_normalization(all_pre50, raw_data["max_min"][0], raw_data["max_min"][1])
final_pred75 = Inverse_normalization(all_pre75, raw_data["max_min"][0], raw_data["max_min"][1])
final_pred90 = Inverse_normalization(all_pre90, raw_data["max_min"][0], raw_data["max_min"][1])
final_target = Inverse_normalization(all_true,  raw_data["max_min"][0], raw_data["max_min"][1])

mae,mape,rmse = masked_mae(final_pred25, final_target,0.0), masked_mape(final_pred25, final_target,0.0)*100,masked_rmse(final_pred25, final_target,0.0)
print('The metrics of student when missing rate is 25%, \nRMSE: {:02.4f}, MAPE: {:02.4f}, MAE: {:02.4f}'.format(rmse, mape, mae))

mae,mape,rmse = masked_mae(final_pred50, final_target,0.0), masked_mape(final_pred50, final_target,0.0)*100,masked_rmse(final_pred50, final_target,0.0)
print('The metrics of student when missing rate is 50%, \nRMSE: {:02.4f}, MAPE: {:02.4f}, MAE: {:02.4f}'.format(rmse, mape, mae))

mae,mape,rmse = masked_mae(final_pred75, final_target,0.0), masked_mape(final_pred75, final_target,0.0)*100,masked_rmse(final_pred75, final_target,0.0)
print('The metrics of student when missing rate is 75%, \nRMSE: {:02.4f}, MAPE: {:02.4f}, MAE: {:02.4f}'.format(rmse, mape, mae))

mae,mape,rmse = masked_mae(final_pred90, final_target,0.0), masked_mape(final_pred90, final_target,0.0)*100,masked_rmse(final_pred90, final_target,0.0)
print('The metrics of student when missing rate is 90%, \nRMSE: {:02.4f}, MAPE: {:02.4f}, MAE: {:02.4f}'.format(rmse, mape, mae))