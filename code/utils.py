import numpy as np
import pandas as pd
import torch
import argparse
from sklearn.preprocessing import minmax_scale, scale
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, auc,recall_score
import scipy.sparse as sp
from matplotlib import pyplot as plt
import scipy.io as sco
import os

def scaley(ymat):
    return (ymat - ymat.min()) / ymat.max()


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def load_data(dis_sim_path, drug_sim_path, drug_dis_path, cuda):
    dis_sim =np.loadtxt(dis_sim_path, delimiter=',')
    drug_sim = np.loadtxt(drug_sim_path, delimiter=',')
    drug_dis = np.loadtxt(drug_dis_path, delimiter=',')
    dis_sim,drug_sim=dis_sim+np.eye(dis_sim.shape[0]),drug_sim+np.eye(drug_sim.shape[0])
    dis_sim = torch.from_numpy(dis_sim).float()
    drug_dis = torch.from_numpy(drug_dis).float()
    drug_sim = torch.from_numpy(drug_sim).float()
    g_drug = norm_adj(drug_sim)
    g_dis = norm_adj(dis_sim.T)
    if cuda:
        dis_sim = dis_sim.cuda()
        drug_dis = drug_dis.cuda()
        drug_sim = drug_sim.cuda()
        g_drug = g_drug.cuda()
        g_dis = g_dis.cuda()
    return dis_sim, drug_dis, drug_sim, g_drug, g_dis
    pass


def neighborhood(feat, k):
    featprod = np.dot(feat.T, feat)
    '''np.diag(featprod):以一维数组的形式返回featprod的对角线元素，
    然后在x维度上重复feat.shape[1]次
    '''
    smat = np.tile(np.diag(featprod), (feat.shape[1], 1))
    dmat = smat + smat.T - 2 * featprod
    dsort = np.argsort(dmat)[:, 1:k + 1]
    C = np.zeros((feat.shape[1], feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i, j] = 1.0
    return C


def normalized(wmat):
    deg = np.diag(np.sum(wmat, axis=0))
    degpow = np.power(deg, -0.5)
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow, wmat), degpow)
    return W


def norm_adj(feat):
    C = neighborhood(feat.T, k=1)
    norm_adj = normalized(C.T * C + np.eye(C.shape[0]))
    g = torch.from_numpy(norm_adj).float()
    return g


def get_metrics(real_score, predict_score):
    real_score, predict_score = real_score.flatten(), predict_score.flatten()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN
    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    plt.plot(x_ROC,y_ROC)
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])
    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    plt.plot(x_PR,y_PR)
    plt.show()
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    print('*'*10)
    print( ' auc:{:.4f} ,aupr:{:.4f},f1_score:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, specificity:{:.4f}, precision:{:.4f}'.format(auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision))
    return [auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision]

