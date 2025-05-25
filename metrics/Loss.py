import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from .mask_metric import masked_mae, masked_mse


def Merlin_Loss_mutilstage(teacher_fore, teacher_HR, 
                student_fore25,  student_HR25, student_H4CL25,
                student_fore50,  student_HR50, student_H4CL50,
                student_fore75,  student_HR75, student_H4CL75,
                student_fore90,  student_HR90, student_H4CL90,
                label, m, temperature, weight_KD, weight_CL, weight_pre, epoch, epoch_Thre):
    """
    :param teacher_HR: Hidden representation of the teacher model
    :param teacher_fore: Forecasting results of the teacher model
    :param student_HR: Hidden representation of the student model
    :param student_fore: Forecasting results of the student model
    :param student_H4CL: Hidden representation used for contrastive learning.
    :param label: True value
    :temperature: The temperature coefficient of the contrastive loss
    :param m: The number of missing rates
    :param weight_CL: The weight of Knowledge distillation
    :param weight_KD: The weight of contrastive loss
    :param epoch: The train epoch
    :return: Loss
    """
    batch_size,_,_ = label.shape 

    L_HD = masked_mse(student_HR25, teacher_HR) + masked_mse(student_HR50, teacher_HR) + masked_mse(student_HR75, teacher_HR) + masked_mse(student_HR90, teacher_HR)

    L_RD = masked_mse(student_fore25, teacher_fore, 0.0) + masked_mse(student_fore50, teacher_fore, 0.0) + masked_mse(student_fore75, teacher_fore, 0.0) + masked_mse(student_fore90, teacher_fore, 0.0)

    L_pre = masked_mae(student_fore25, label, 0.0) + masked_mae(student_fore50, label, 0.0) + masked_mae(student_fore75, label, 0.0) + masked_mae(student_fore90, label, 0.0)

    if epoch <= epoch_Thre:
        L_CL = CL_loss(student_H4CL25, student_H4CL50, temperature) + CL_loss(student_H4CL25, student_H4CL75, temperature) + CL_loss(student_H4CL25, student_H4CL90, temperature) + CL_loss(student_H4CL50, student_H4CL75, temperature) + CL_loss(student_H4CL50, student_H4CL90, temperature) + CL_loss(student_H4CL75, student_H4CL90, temperature)
        L_CL = 2 * L_CL/(m*(m-1))
        L_final = weight_pre * L_pre + weight_KD * (L_HD + L_RD) + weight_CL *  L_CL
    else:
        weight_KD = weight_KD / (epoch - epoch_Thre)
        L_final = weight_pre * L_pre + weight_KD * (L_HD + L_RD)

    return L_final


def Merlin_Loss_add(teacher_fore, teacher_HR, 
                student_fore25,  student_HR25, student_H4CL25,
                student_fore50,  student_HR50, student_H4CL50,
                student_fore75,  student_HR75, student_H4CL75,
                student_fore90,  student_HR90, student_H4CL90,
                label, m, temperature, weight_KD, weight_CL, weight_pre, epoch, epoch_Thre):
    """
    :param teacher_HR: Hidden representation of the teacher model
    :param teacher_fore: Forecasting results of the teacher model
    :param student_HR: Hidden representation of the student model
    :param student_fore: Forecasting results of the student model
    :param student_H4CL: Hidden representation used for contrastive learning.
    :param label: True value
    :temperature: The temperature coefficient of the contrastive loss
    :param m: The number of missing rates
    :param weight_CL: The weight of Knowledge distillation
    :param weight_KD: The weight of contrastive loss
    :param epoch: The train epoch
    :return: Loss
    """
    batch_size,_,_ = label.shape 

    L_HD = masked_mse(student_HR25, teacher_HR) + masked_mse(student_HR50, teacher_HR) + masked_mse(student_HR75, teacher_HR) + masked_mse(student_HR90, teacher_HR)

    L_RD = masked_mse(student_fore25, teacher_fore, 0.0) + masked_mse(student_fore50, teacher_fore, 0.0) + masked_mse(student_fore75, teacher_fore, 0.0) + masked_mse(student_fore90, teacher_fore, 0.0)

    L_pre = masked_mae(student_fore25, label, 0.0) + masked_mae(student_fore50, label, 0.0) + masked_mae(student_fore75, label, 0.0) + masked_mae(student_fore90, label, 0.0)
    
    L_CL = CL_loss(student_H4CL25, student_H4CL50, temperature) + CL_loss(student_H4CL25, student_H4CL75, temperature) + CL_loss(student_H4CL25, student_H4CL90, temperature) + CL_loss(student_H4CL50, student_H4CL75, temperature) + CL_loss(student_H4CL50, student_H4CL90, temperature) + CL_loss(student_H4CL75, student_H4CL90, temperature)
    
    L_CL = 2 * L_CL/(m*(m-1))

    if epoch <= epoch_Thre:
        L_final = weight_pre * L_pre + weight_KD * (L_HD + L_RD) + weight_CL *  L_CL
    else:
        weight_KD = weight_KD / (epoch - epoch_Thre)
        weight_CL = weight_CL / (epoch - epoch_Thre)
        L_final = weight_pre * L_pre + weight_KD * (L_HD + L_RD) + weight_CL *  L_CL

    return L_final

def Loss_KD_off(teacher_fore, teacher_HR, 
                student_fore25,  student_HR25, student_H4CL25,
                student_fore50,  student_HR50, student_H4CL50,
                student_fore75,  student_HR75, student_H4CL75,
                student_fore90,  student_HR90, student_H4CL90,
                label, m, weight_KD, weight_pre):
    """
    :param teacher_HR: Hidden representation of the teacher model
    :param teacher_fore: Forecasting results of the teacher model
    :param student_HR: Hidden representation of the student model
    :param student_fore: Forecasting results of the student model
    :param student_H4CL: Hidden representation used for contrastive learning.
    :param label: True value
    :param m: The number of missing rates
    :param weight: The weight of Knowledge distillation
    :return: Loss
    """

    L_HD = masked_mse(student_HR25, teacher_HR) + masked_mse(student_HR50, teacher_HR) + masked_mse(student_HR75, teacher_HR) + masked_mse(student_HR90, teacher_HR)

    L_RD = masked_mse(student_fore25, teacher_fore, 0.0) + masked_mse(student_fore50, teacher_fore, 0.0) + masked_mse(student_fore75, teacher_fore, 0.0) + masked_mse(student_fore90, teacher_fore, 0.0)

    L_pre = masked_mae(student_fore25, label, 0.0) + masked_mae(student_fore50, label, 0.0) + masked_mae(student_fore75, label, 0.0) + masked_mae(student_fore90, label, 0.0)

    L_final = weight_pre * L_pre + weight_KD * (L_HD + L_RD)

    return L_final


def Loss_KD_off_self(teacher_fore, teacher_HR, 
                student_fore25,  student_HR25, student_H4CL25,
                student_fore50,  student_HR50, student_H4CL50,
                student_fore75,  student_HR75, student_H4CL75,
                student_fore90,  student_HR90, student_H4CL90,
                label, m, weight_off, weight_self, weight_pre):
    """
    :param teacher_HR: Hidden representation of the teacher model
    :param teacher_fore: Forecasting results of the teacher model
    :param student_HR: Hidden representation of the student model
    :param student_fore: Forecasting results of the student model
    :param m: The number of missing rates
    :param weight: The weight of loss function
    :return: Loss
    """

    L_HD = masked_mse(student_HR25, teacher_HR) + masked_mse(student_HR50, teacher_HR) + masked_mse(student_HR75, teacher_HR) + masked_mse(student_HR90, teacher_HR)

    L_RD = masked_mse(student_fore25, teacher_fore, 0.0) + masked_mse(student_fore50, teacher_fore, 0.0) + masked_mse(student_fore75, teacher_fore, 0.0) + masked_mse(student_fore90, teacher_fore, 0.0)

    L_pre = masked_mae(student_fore25, label, 0.0) + masked_mae(student_fore50, label, 0.0) + masked_mae(student_fore75, label, 0.0) + masked_mae(student_fore90, label, 0.0)

    L_SELF_HD = masked_mse(student_HR25, student_HR50) + masked_mse(student_HR25, student_HR75) + masked_mse(student_HR25, student_HR90) + masked_mse(student_HR50, student_HR75) + masked_mse(student_HR50, student_HR90) + masked_mse(student_HR75, student_HR90)

    L_SELF_RD = masked_mse(student_fore25, student_fore50, 0.0) + masked_mse(student_fore25, student_fore75, 0.0) + masked_mse(student_fore25, student_fore90, 0.0) + masked_mse(student_fore50, student_fore75, 0.0) + masked_mse(student_fore50, student_fore90, 0.0) + masked_mse(student_fore75, student_fore90, 0.0)

    L_SELF_HD = 2 * L_SELF_HD/(m*(m-1))

    L_SELF_RD = 2 * L_SELF_RD/(m*(m-1))

    L_final = weight_pre * L_pre + weight_off * (L_HD + L_RD) + weight_self * (L_SELF_HD + L_SELF_RD)

    return L_final


def CL_loss(z1, z2, temperature):
    B = z1.size(0)

    z1 = z1.reshape(B,-1)
    z2 = z2.reshape(B,-1)

    # normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # concat: [2B, D]
    z = torch.cat([z1, z2], dim=0)  # shape: [2B, D]

    # Calculate similarity [2B, 2B]
    sim = torch.matmul(z, z.T) / temperature

    # label：positive lable between i <-> i + B 
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels], dim=0)  # [2B]

    # mask itself
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))

    # CrossEntropyLoss
    loss = F.cross_entropy(sim, labels)
    return loss


def CL_loss2(z1, z2, temperature):
    B = z1.size(0)

    z1 = z1.reshape(B,-1)
    z2 = z2.reshape(B,-1)

    # normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # concat: [2B, D]
    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    # Calculate similarity [2B, 2B]
    sim = torch.matmul(z, z.T) / temperature  # each sim[i, j] is the similarity betwween z[i] with z[j]

    # mask itself
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)

    # obtain positive sample index: for ith sample，its positive sample is i ± B
    positives = torch.cat([
        torch.arange(B, 2 * B),
        torch.arange(0, B)
    ]).to(z.device)  # [2B]

    # Calculate log-softmax
    sim_log_softmax = F.log_softmax(sim, dim=1)

    # obtain the log-softmax value in Positive sample
    loss = -sim_log_softmax[torch.arange(2 * B), positives]

    return loss.mean()

def multi_view_CL_loss(z_list, temperature=0.5):
    """
    z_list: List of tensors [z1, z2, ..., zn], each with shape [B, D]
    temperature: float, temperature scaling
    """
    N = len(z_list)  # number of missing rates
    B = z_list[0].shape[0]

    z_all = [z.view(B, -1) for z in z_list]

    # Step 1: concat: [N*B, D]
    z_all = torch.cat(z_all, dim=0)  # [N*B, D]
    z_all = F.normalize(z_all, dim=1)

    # Step 2: Calculate similarity [N*B, N*B]
    sim = torch.matmul(z_all, z_all.T) / temperature

    # Step 3: mask itself
    mask = torch.eye(N * B, device=sim.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Step 4: btain positive sample index
    labels = torch.arange(B, device=sim.device)
    positives = []
    for i in range(N):
        for j in range(N):
            if i != j:
                anchor_idx = labels + i * B
                pos_idx = labels + j * B
                positives.append((anchor_idx, pos_idx))

    # Step 5: Calculate cl loss for all missing rates
    loss = 0.0
    for anchor_idx, pos_idx in positives:
        sim_pair = sim[anchor_idx]  # [B, N*B]
        pos_sim = sim_pair.gather(1, pos_idx.unsqueeze(1)) 
        loss_i = -torch.log(pos_sim.squeeze(1) / sim_pair.exp().sum(dim=1))
        loss += loss_i.mean()

    # each anchor have (N - 1) Positive data pairs
    return loss / (N * (N - 1))
