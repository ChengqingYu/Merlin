import os
import sys
import copy
import random
import csv

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd


# random seed
seed = 3407
random.seed(seed)
np.random.seed(seed)

# Sliding window size. Data set division ratio
history_seq_len = 12 # History length
future_seq_len = 12 # Future length
train_ratio = 0.6
valid_ratio = 0.2
target_channel = 0 # target channel(s)

# data path
data_line = "datasets/"
data_name = "PEMS04"
data_file_path = data_line + data_name + "/" + data_name + ".npz"

data = np.load(data_file_path)["data"]
data = data[..., target_channel]
print("raw time series shape: {0}".format(data.shape))

###The number of samples after data division
l, n = data.shape
num_samples = data.shape[0] - history_seq_len - future_seq_len + 1
train_num_short = round(num_samples * train_ratio)
valid_num_short = round(num_samples * valid_ratio)
test_num_short = num_samples - train_num_short - valid_num_short

print("number of training samples:{0}".format(train_num_short))
print("number of validation samples:{0}".format(valid_num_short))
print("number of test samples:{0}".format(test_num_short))

###data normalize
def normalize(x,max_data,min_data):
    return (x - min_data) / (max_data - min_data)

max_data,min_data = data.max(), data.min()
max_min = [max_data, min_data]
max_min = np.array(max_min)

data_new = normalize(data,max_data,min_data)

### Sliding window
def feature_target(data,input_len,output_len):
    fin_feature = []
    fin_target = []
    data_len = data.shape[0]
    for i in range(data_len-input_len - output_len + 1):
        lin_fea_seq = data[i:i+input_len,:]
        lin_tar_seq = data[i+input_len:i+input_len + output_len,:]
        fin_feature.append(lin_fea_seq)
        fin_target.append(lin_tar_seq)
    fin_feature = np.array(fin_feature)
    fin_target = np.array(fin_target)
    return fin_feature, fin_target

raw_feature, raw_target = feature_target(data_new ,history_seq_len, future_seq_len)


print("----------------Sliding window-------------------")
### train
train_x = raw_feature[0:train_num_short,:,:]
train_y = raw_target[0:train_num_short,:,:]
### Validation
vail_x = raw_feature[train_num_short:train_num_short+valid_num_short,:,:]
vail_y = raw_target[train_num_short:train_num_short+valid_num_short,:,:]
### test
test_x = raw_feature[train_num_short+valid_num_short:,:,:]
test_y = raw_target[train_num_short+valid_num_short:,:,:]

print(train_x.shape)
print(train_y.shape)
print(vail_x.shape)
print(vail_y.shape)
print(test_x.shape)
print(test_y.shape)

print("----------------mask_matric-------------------")

def random_mask_matric_all(data):
    # Generate the mask matrix with a missing rate of mask_percentage.
    masked_matrix25 = []
    masked_matrix50 = []
    masked_matrix75 = []
    masked_matrix90 = []
    # Calculate the number of points to be masked.
    L, N = data.shape

    for i in range(N):
        
        num_mask_points_25 = int(L * 0.25)
        num_mask_points_50 = int(L * 0.5)
        num_mask_points_75 = int(L * 0.75)
        num_mask_points_90 = int(L * 0.9)

        # Missing rate 90%
        # Generate random indices.
        mask_indices_90 = np.random.choice(L, size = num_mask_points_90, replace=False)
        mask_indices_90 = mask_indices_90.tolist()
        # Generate a matrix filled with ones.
        matrix_90 = np.ones((L))
        # Set the values at the corresponding positions to 0.
        matrix_90[mask_indices_90] = 0
        masked_matrix90.append(matrix_90)

        # missing rates 75%
        selected_indices75 = random.sample(range(len(mask_indices_90)), num_mask_points_75)
        selected_indices75.sort()

        mask_indices_75 = [mask_indices_90[i] for i in selected_indices75]
        matrix_75 = np.ones((L))
        matrix_75[mask_indices_75] = 0
        masked_matrix75.append(matrix_75)

        # missing rates 50%
        selected_indices50 = random.sample(range(len(mask_indices_75)), num_mask_points_50)
        selected_indices50.sort()

        mask_indices_50 = [mask_indices_75[i] for i in selected_indices50]
        matrix_50 = np.ones((L))
        matrix_50[mask_indices_50] = 0
        masked_matrix50.append(matrix_50)

        # missing rates 25%
        selected_indices25 = random.sample(range(len(mask_indices_50)), num_mask_points_25)
        selected_indices25.sort()

        mask_indices_25 = [mask_indices_50[i] for i in selected_indices25]
        matrix_25 = np.ones((L))
        matrix_25[mask_indices_25] = 0
        masked_matrix25.append(matrix_25)


    masked_matrix25 = np.array(masked_matrix25)
    masked_matrix25 = masked_matrix25.transpose((1,0))

    masked_matrix50 = np.array(masked_matrix50)
    masked_matrix50 = masked_matrix50.transpose((1,0))

    masked_matrix75 = np.array(masked_matrix75)
    masked_matrix75 = masked_matrix75.transpose((1,0))

    masked_matrix90 = np.array(masked_matrix90)
    masked_matrix90 = masked_matrix90.transpose((1,0))
    return masked_matrix25, masked_matrix50, masked_matrix75, masked_matrix90

matric_25, matric_50 ,matric_75 ,matric_90 = random_mask_matric_all(data_new)

matric_25,_ = feature_target(matric_25,history_seq_len, future_seq_len)
matric_50,_ = feature_target(matric_50,history_seq_len, future_seq_len)
matric_75,_ = feature_target(matric_75,history_seq_len, future_seq_len)
matric_90,_ = feature_target(matric_90,history_seq_len, future_seq_len)

train_matric_25 = matric_25[0:train_num_short,:,:]
vaild_matric_25 = matric_25[train_num_short:train_num_short+valid_num_short,:,:]
test_matric_25 = matric_25[train_num_short+valid_num_short:,:,:]

train_matric_50 = matric_50[0:train_num_short,:,:]
vaild_matric_50 = matric_50[train_num_short:train_num_short+valid_num_short,:,:]
test_matric_50 = matric_50[train_num_short+valid_num_short:,:,:]

train_matric_75 = matric_75[0:train_num_short,:,:]
vaild_matric_75 = matric_75[train_num_short:train_num_short+valid_num_short,:,:]
test_matric_75 = matric_75[train_num_short+valid_num_short:,:,:]

train_matric_90 = matric_90[0:train_num_short,:,:]
vaild_matric_90 = matric_90[train_num_short:train_num_short+valid_num_short,:,:]
test_matric_90 = matric_90[train_num_short+valid_num_short:,:,:]


print(train_matric_25.shape)
print(vaild_matric_25.shape)
print(test_matric_25.shape)


print("----------------time_embedding-------------------")
steps_per_day = 288

###Time in day embedding
tod = [i % steps_per_day /
       steps_per_day for i in range(data_new.shape[0])]
tod = np.array(tod)
tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
tod_tiled,tod_target = feature_target(tod_tiled[:,:,-1] ,history_seq_len, future_seq_len)
tod_tiled = np.expand_dims(tod_tiled,axis=-1)
tod_target = np.expand_dims(tod_target,axis=-1)

### train
train_tod_tiled = tod_tiled[0:train_num_short,:,:,:]
train_tod_target = tod_target[0:train_num_short,:,:,:]
### Validation
vail_tod_tiled  = tod_tiled[train_num_short:train_num_short+valid_num_short,:,:,:]
vail_tod_target = tod_target[train_num_short:train_num_short+valid_num_short,:,:,:]
### test
test_tod_tiled  = tod_tiled[train_num_short+valid_num_short:,:,:,:]
test_tod_target = tod_target[train_num_short+valid_num_short:,:,:,:]

### day in week embedding
dow = [(i // steps_per_day) % 7 / 7 for i in range(data_new.shape[0])]
dow = np.array(dow)
dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
dow_tiled, dow_target = feature_target(dow_tiled[:,:,-1] ,history_seq_len, future_seq_len)
dow_tiled = np.expand_dims(dow_tiled,axis=-1)
dow_target = np.expand_dims(dow_target,axis=-1)

### train
train_dow_tiled = dow_tiled[0:train_num_short,:,:,:]
train_dow_target = dow_target[0:train_num_short,:,:,:]
### Validation
vail_dow_tiled = dow_tiled[train_num_short:train_num_short+valid_num_short,:,:,:]
vail_dow_target = dow_target[train_num_short:train_num_short+valid_num_short,:,:,:]
### test
test_dow_tiled = dow_tiled[train_num_short+valid_num_short:,:,:,:]
test_dow_target = dow_target[train_num_short+valid_num_short:,:,:,:]


### concatenate

print("----------------finall_data-------------------")
### train
train_x = np.expand_dims(train_x,axis=-1)
train_x = np.concatenate((train_x, train_tod_tiled, train_dow_tiled), axis=-1)

train_y = np.expand_dims(train_y,axis=-1)
train_y = np.concatenate((train_y, train_tod_target, train_dow_target), axis=-1)

### Validation
vail_x = np.expand_dims(vail_x,axis=-1)
vail_x = np.concatenate((vail_x, vail_tod_tiled, vail_dow_tiled), axis=-1)

vail_y = np.expand_dims(vail_y,axis=-1)
vail_y = np.concatenate((vail_y, vail_tod_target, vail_dow_target), axis=-1)

### test
test_x = np.expand_dims(test_x,axis=-1)
test_x = np.concatenate((test_x, test_tod_tiled, test_dow_tiled), axis=-1)

test_y = np.expand_dims(test_y,axis=-1)
test_y = np.concatenate((test_y, test_tod_target, test_dow_target), axis=-1)

print(train_x.shape)
print(train_y.shape)
print(vail_x.shape)
print(vail_y.shape)
print(test_x.shape)
print(test_y.shape)

np.savez(data_line + data_name + "/data"+ str(history_seq_len) + ".npz",
        train_x = train_x,
        train_y = train_y,
        vail_x  = vail_x,
        vail_y  = vail_y,
        test_x  = test_x,
        test_y  = test_y,

        train_matric_25 = train_matric_25,
        vaild_matric_25 = vaild_matric_25,
        test_matric_25 = test_matric_25,

        train_matric_50 = train_matric_50,
        vaild_matric_50 = vaild_matric_50,
        test_matric_50 = test_matric_50,

        train_matric_75 = train_matric_75,
        vaild_matric_75 = vaild_matric_75,
        test_matric_75 = test_matric_75,

        train_matric_90 = train_matric_90,
        vaild_matric_90 = vaild_matric_90,
        test_matric_90 = test_matric_90,
        max_min = max_min
         )