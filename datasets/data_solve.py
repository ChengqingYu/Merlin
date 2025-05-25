import numpy as np
import pandas as pd
import copy
import random
import csv
import torch


def random_mask_matric_history_observe(data, mask_percentage):
    # Generate the mask matrix with a missing rate of mask_percentage.
    
    # Calculate the number of points to be masked.
    B, L, N = data.shape
    num_mask_points = int(N * L * mask_percentage)
    # Generate random indices.
    mask_indices = np.random.choice(N * L, size=num_mask_points, replace=False)
    # Generate a matrix filled with ones.
    matrix = np.ones((L, N))
    # Set the values at the corresponding positions to 0.
    matrix_flattened = matrix.reshape(N*L)
    matrix_flattened[mask_indices] = 0
    # Reshape the tensor to its original shape.
    masked_matrix = matrix_flattened.reshape(L, N)
    return masked_matrix


def random_mask_matric_all_radom(data, mask_percentage):
    # Generate the mask matrix with a missing rate of mask_percentage.
    masked_matrix = []
    # Calculate the number of points to be masked.
    L, N = data.shape

    for i in range(N):
        num_mask_points = int(L * mask_percentage)
        # Generate random indices.
        mask_indices = np.random.choice(L, size=num_mask_points, replace=False)
        # Generate a matrix filled with ones.
        matrix = np.ones((L))
        # Set the values at the corresponding positions to 0.
        matrix[mask_indices] = 0
        masked_matrix.append(matrix)
    masked_matrix = np.array(masked_matrix)
    masked_matrix = masked_matrix.transpose((1,0))
    return masked_matrix


def random_mask_matric_all_subset(data):
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


def tensor_mask(data, masked_matrix):
    feature = data[:,:,:,0]
    time_line = data[:,:,:,1:]

    feature = feature * masked_matrix
    feature = np.expand_dims(feature,axis=-1)
    feature = np.concatenate((feature,time_line),axis=-1)

    return feature


def batch_data_solve_all_mask(data, data_index, data_name, device):

    if data_name == "vaild":
        feature = data["vail_x"][data_index,:,:,:]
        target = data["vail_y"][data_index,:,:,:]

        ### Obtain the masked features.
        feature_25 = tensor_mask(feature, data["vaild_matric_25"][data_index,:,:])
        feature_50 = tensor_mask(feature, data["vaild_matric_50"][data_index,:,:])
        feature_75 = tensor_mask(feature, data["vaild_matric_75"][data_index,:,:])
        feature_90 = tensor_mask(feature, data["vaild_matric_90"][data_index,:,:])

    elif data_name == "train":
        feature = data["train_x"][data_index,:,:,:]
        target = data["train_y"][data_index,:,:,:]

        ### Obtain the masked features.
        feature_25 = tensor_mask(feature, data["train_matric_25"][data_index,:,:])
        feature_50 = tensor_mask(feature, data["train_matric_50"][data_index,:,:])
        feature_75 = tensor_mask(feature, data["train_matric_75"][data_index,:,:])
        feature_90 = tensor_mask(feature, data["train_matric_90"][data_index,:,:])

    else:
        feature = data["test_x"][data_index,:,:,:]
        target = data["test_y"][data_index,:,:,:]

        ### Obtain the masked features.
        feature_25 = tensor_mask(feature, data["test_matric_25"][data_index,:,:])
        feature_50 = tensor_mask(feature, data["test_matric_50"][data_index,:,:])
        feature_75 = tensor_mask(feature, data["test_matric_75"][data_index,:,:])
        feature_90 = tensor_mask(feature, data["test_matric_90"][data_index,:,:])

    ### Convert to a torch tensor and move it to the specified device.
    feature = torch.tensor(feature).to(torch.float32).to(device)
    target = torch.tensor(target).to(torch.float32).to(device)
    
    feature_25 = torch.tensor(feature_25).to(torch.float32).to(device)
    feature_50 = torch.tensor(feature_50).to(torch.float32).to(device)
    feature_75 = torch.tensor(feature_75).to(torch.float32).to(device)
    feature_90 = torch.tensor(feature_90).to(torch.float32).to(device)

    feature_mask = torch.cat([feature_25,feature_50,feature_75,feature_90], dim = 0)

    return feature, target, feature_mask


def batch_data_solve_one_mask(data, data_index, data_name, mask_rate, device):
    if data_name == "vaild":
        feature = data["vail_x"][data_index,:,:,:]
        target = data["vail_y"][data_index,:,:,:]

        ### Obtain the masked features.
        if mask_rate == 0.25:
            feature_mask = tensor_mask(feature, data["vaild_matric_25"][data_index,:,:])
        elif mask_rate == 0.5:   
            feature_mask = tensor_mask(feature, data["vaild_matric_50"][data_index,:,:])
        elif mask_rate == 0.75:    
            feature_mask = tensor_mask(feature, data["vaild_matric_75"][data_index,:,:])
        else:
            feature_mask = tensor_mask(feature, data["vaild_matric_90"][data_index,:,:])

    elif data_name == "train":
        feature = data["train_x"][data_index,:,:,:]
        target = data["train_y"][data_index,:,:,:]

        ### Obtain the masked features.
        if mask_rate == 0.25:
            feature_mask = tensor_mask(feature, data["train_matric_25"][data_index,:,:])
        elif mask_rate == 0.5:   
            feature_mask = tensor_mask(feature, data["train_matric_50"][data_index,:,:])
        elif mask_rate == 0.75:    
            feature_mask = tensor_mask(feature, data["train_matric_75"][data_index,:,:])
        else:
            feature_mask = tensor_mask(feature, data["train_matric_90"][data_index,:,:])


    else:
        feature = data["test_x"][data_index,:,:,:]
        target = data["test_y"][data_index,:,:,:]

        ### Obtain the masked features.
        if mask_rate == 0.25:
            feature_mask = tensor_mask(feature, data["test_matric_25"][data_index,:,:])
        elif mask_rate == 0.5:   
            feature_mask = tensor_mask(feature, data["test_matric_50"][data_index,:,:])
        elif mask_rate == 0.75:    
            feature_mask = tensor_mask(feature, data["test_matric_75"][data_index,:,:])
        else:
            feature_mask = tensor_mask(feature, data["test_matric_90"][data_index,:,:])

    ### Convert to a torch tensor and move it to the specified device.
    feature = torch.tensor(feature).to(torch.float32).to(device)
    target = torch.tensor(target).to(torch.float32).to(device)
    feature_mask = torch.tensor(feature_mask).to(torch.float32).to(device)

    return feature, target, feature_mask


def batch_data_solve_one(feature, target, mask_matric, data_index, device):

    feature = feature[data_index,:,:,:]
    target = target[data_index,:,:,:]
    mask_matric = mask_matric[data_index,:,:]
    feature_mask = tensor_mask(feature, mask_matric)

    feature = torch.tensor(feature).to(torch.float32).to(device)
    target = torch.tensor(target).to(torch.float32).to(device)
    feature_mask = torch.tensor(feature_mask).to(torch.float32).to(device)

    return feature, target, feature_mask


def batch_data_solve_teacher(feature, target, data_index, device):

    feature = feature[data_index,:,:,:]
    target = target[data_index,:,:,:]

    feature = torch.tensor(feature).to(torch.float32).to(device)
    target = torch.tensor(target).to(torch.float32).to(device)

    return feature, target


def batch_data_solve_student(feature, target, mask_25, mask_50, mask_75, mask_90, data_index, device):


    feature = feature[data_index,:,:,:]
    target = target[data_index,:,:,:]

    ### Obtain the masked features.
    
    feature_25 = tensor_mask(feature, mask_25[data_index,:,:])
    feature_50 = tensor_mask(feature, mask_50[data_index,:,:])
    feature_75 = tensor_mask(feature, mask_75[data_index,:,:])
    feature_90 = tensor_mask(feature, mask_90[data_index,:,:])
    """
    feature_25 = tensor_mask(feature, mask_25)
    feature_50 = tensor_mask(feature, mask_50)
    feature_75 = tensor_mask(feature, mask_75)
    feature_90 = tensor_mask(feature, mask_90)
    """
    ### Convert to a torch tensor and move it to the specified device.
    feature = torch.tensor(feature).to(torch.float32).to(device)
    target = torch.tensor(target).to(torch.float32).to(device)
    
    feature_25 = torch.tensor(feature_25).to(torch.float32).to(device)
    feature_50 = torch.tensor(feature_50).to(torch.float32).to(device)
    feature_75 = torch.tensor(feature_75).to(torch.float32).to(device)
    feature_90 = torch.tensor(feature_90).to(torch.float32).to(device)

    # feature_mask = torch.cat([feature_25,feature_50,feature_75,feature_90], dim = 0)

    return feature, target, feature_25, feature_50, feature_75, feature_90