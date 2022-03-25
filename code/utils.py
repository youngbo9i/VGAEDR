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

def load_DRIMC(root_dir="../Dataset1/DRHGCN/drimc", name="lrssl", reduce=True):
    """ C drug:658, disease:409 association:2520 (False 2353)
        PREDICT drug:593, disease:313 association:1933 (Fdataset)
        LRSSL drug: 763, disease:681, association:3051
    """
    #药物化学相似度
    drug_chemical = pd.read_csv(os.path.join(root_dir, f"{name}_simmat_dc_chemical.txt"), sep="\t", index_col=0)
    #药物相似性矩阵
    drug_domain = pd.read_csv(os.path.join(root_dir, f"{name}_simmat_dc_domain.txt"), sep="\t", index_col=0)
    #药物基因相似度
    drug_go = pd.read_csv(os.path.join(root_dir, f"{name}_simmat_dc_go.txt"), sep="\t", index_col=0)
    #疾病相似度
    disease_sim = pd.read_csv(os.path.join(root_dir, f"{name}_simmat_dg.txt"), sep="\t", index_col=0)
    if reduce:
        #取这三种药物相似度的平均值
        drug_sim =  (drug_chemical+drug_domain+drug_go)/3
    else:
        drug_sim = drug_chemical
    #药物疾病关联矩阵
    drug_disease = pd.read_csv(os.path.join(root_dir, f"{name}_admat_dgc.txt"), sep="\t", index_col=0).T
    if name=="lrssl":
        #观察数据知道这两类数据集的药物-疾病关联矩阵行列相反
        drug_disease = drug_disease.T
    #将药物相似度的Dataframe类型转换成numpy
    rr = drug_sim.to_numpy(dtype=np.float32)
    # print('药物smi:',rr,rr.shape)
    #转换药物疾病关联矩阵
    rd = drug_disease.to_numpy(dtype=np.float32)

    #转换疾病相似度矩阵
    dd = disease_sim.to_numpy(dtype=np.float32)
    # print('疾病smi',dd,dd.shape)
    # print('关联矩阵:')
    #获得药物名
    rname = drug_sim.columns.to_numpy()
    #获得疾病名
    dname = disease_sim.columns.to_numpy()
    return {"drug":rr,
            "disease":dd,
            "Wrname":rname,
            "Wdname":dname,
            "didr":rd.T}
def load_data(dis_sim_path, drug_sim_path, drug_dis_path, cuda):
    # gti相当于是disese feature
    # dis_sim = pd.read_csv(dis_sim_path, header=None)
    # drug_sim = pd.read_csv(drug_sim_path, header=None)
    # drug_dis = pd.read_csv(drug_dis_path, header=None)
    #LAGCN
    dis_sim =np.loadtxt(dis_sim_path, delimiter=',')
    drug_sim = np.loadtxt(drug_sim_path, delimiter=',')
    drug_dis = np.loadtxt(drug_dis_path, delimiter=',')
    #DRHGCN
    ## Fdataset(使用mse，k=0,10择）
    # fdataset=sio.loadmat('../Dataset1/DRHGCN/Fdataset.mat')
    # dis_sim =fdataset['disease']
    # drug_sim = fdataset['drug']
    # drug_dis = fdataset['didr'].T
    ## Cdataset(使用mse，k=0,10择）
    # Cdataset=sio.loadmat('../Dataset1/DRHGCN/Cdataset.mat')
    # dis_sim =Cdataset['disease']
    # drug_sim = Cdataset['drug']
    # drug_dis = Cdataset['didr'].T
    ##lrssl(疾病使用交叉熵)
    # lrssl_data=load_DRIMC()
    # drug_sim,dis_sim,drug_dis=lrssl_data['drug'],lrssl_data['disease'],lrssl_data['didr'].T
    # dis_sim, drug_sim, drug_dis = np.array(dis_sim), np.array(drug_sim), np.array(drug_dis)
    ############################################
    dis_sim = torch.from_numpy(dis_sim).float()
    drug_dis = torch.from_numpy(drug_dis).float()
    drug_sim = torch.from_numpy(drug_sim).float()
    #得到药物的领接矩阵(最近的k个赋值为1，且标准化领接矩阵,就是d-1/2 *A *d1/2)
    g_drug = norm_adj(drug_sim)
    g_dis = norm_adj(dis_sim.T)
    # print(gl,gd)
    if cuda:
        dis_sim = dis_sim.cuda()
        drug_dis = drug_dis.cuda()
        drug_sim = drug_sim.cuda()
        g_drug = g_drug.cuda()
        g_dis = g_dis.cuda()
    # print('ldit,ldit.shape',ldit,ldit.shape)
    return dis_sim, drug_dis, drug_sim, g_drug, g_dis
    pass


def neighborhood(feat, k):
    # compute C，计算公式（16）中的C(有时间考虑用余弦相似度来定义距离)
    #对feat做矩阵乘法(元素ij相当于向量i和向量j的
    featprod = np.dot(feat.T, feat)
    '''np.diag(featprod):以一维数组的形式返回featprod的对角线元素，
    然后在x维度上重复feat.shape[1]次
    '''
    smat = np.tile(np.diag(featprod), (feat.shape[1], 1))
    #这里我们单独只看一行，smat+smat.T不会改变smat的大小顺序，然后减去2*featprod，如果featprod比较小
    #那么值就比较大，我们挑选dmat中较大的，就是featprod中较小的k个
    dmat = smat + smat.T - 2 * featprod
    dsort = np.argsort(dmat)[:, 1:k + 1]
    C = np.zeros((feat.shape[1], feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            #与第i个节点最近的k个节点赋值为1
            C[i, j] = 1.0
    return C


def normalized(wmat):
    #求度矩阵
    deg = np.diag(np.sum(wmat, axis=0))
    degpow = np.power(deg, -0.5)
    #0的负0.5次方没意义，赋值为0
    degpow[np.isinf(degpow)] = 0
    #这里就是标准化矩阵的公式
    W = np.dot(np.dot(degpow, wmat), degpow)
    return W


def norm_adj(feat):
    #计算公式（16）中的C,这里原来的k=10,这里我改成
    C = neighborhood(feat.T, k=1)
    #计算公式（16）中的A_hat
    norm_adj = normalized(C.T * C + np.eye(C.shape[0]))
    g = torch.from_numpy(norm_adj).float()
    return g


def show_auc(y_true,ymat):
    # path = 'Dataset' + str(data)
    # ldi = pd.read_csv(path + '/drug_dis.csv', header=None)
    # ldi = np.array(ldi)
    y_true = y_true.flatten()
    # print(ymat.shape)
    ymat = ymat.flatten()
    # print(ymat)
    fpr, tpr, rocth = roc_curve(y_true, ymat)
    # print('recall:{:.4f}'.format(tpr))
    auroc = auc(fpr, tpr)
    # np.savetxt('roc.txt',np.vstack((fpr,tpr)),fmt='%10.5f',delimiter=',')
    precision, recall, prth = precision_recall_curve(y_true, ymat)
    aupr = auc(recall, precision)
    # np.savetxt('pr.txt',np.vstack((recall,precision)),fmt='%10.5f',delimiter=',')
    print('AUROC= %.4f | AUPR= %.4f' % (auroc, aupr))
    # rocdata = np.loadtxt('roc.txt',delimiter=',')
    # prdata = np.loadtxt('pr.txt',delimiter=',')
    # plt.figure()
    # plt.plot(rocdata[0],rocdata[1])
    # plt.plot(prdata[0],prdata[1])
    # plt.show()
    return auroc, aupr


def get_metrics(real_score, predict_score):
    real_score, predict_score = real_score.flatten(), predict_score.flatten()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    # 160373
    sorted_predict_score_num = len(sorted_predict_score)
    # 阈值的选取是从 sorted_predict_score中按从下到大的顺序每间隔1000取一个score作为阈值，一共选取了999个threshold(从小到大)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    # predict_score_matrix 是999*160862,每一行可以根据一个threshold计算出混淆矩阵
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    # predict_score_matrix是999*160862,thresholds转置后是999*1,通过广播机制进行比较，那么，999行的每一行都是基于不同的thresholds判断正负样本
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    # predict_score_matrix(999,160862),到这里一行就是根据一个阈值判断的正负样本。
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    # 999*160862 与16082*1的矩阵做矩阵乘法，得到999*1(999个数字每个数字都是TP的数量，且阈值越大，预测对的正例也多)
    TP = predict_score_matrix.dot(real_score.T)
    # FP=预测的正例数量-TP(预测对的正例数量)(阈值越大，预测对正例(TP)的就也多，相应的fp也就越小）
    FP = predict_score_matrix.sum(axis=1) - TP
    # 实际是正例的-预测对的是正例的数量=被误判为负例的正例（FN）
    FN = real_score.sum() - TP
    # 判断对的负例=样本数-判断对正例-误判为正的负例-误判为负的正例
    TN = len(real_score.T) - TP - FP - FN

    # fpr：负例中被误判为正例的比率(低好)(阈值越大，fpr越小,因为被判断为正例的数量就很少了，判断错的也就少了)
    fpr = FP / (FP + TN)
    # tpr：正例中判断为正例的比率(就是recall，高好)(阈值越大，tpr小，因为本身判断为正的数量也少)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    # 为了让roc曲线从原点开始(将第一个坐标代替为(0,0))
    # print('ROC_dot_matrix.shape:',ROC_dot_matrix.shape)
    ROC_dot_matrix.T[0] = [0, 0]
    # 为了让roc曲线到（1,1）结束，np.c_：按行连接
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T

    plt.plot(x_ROC,y_ROC)
    # 计算无数个梯形面积来拟合pr曲线下的面积 x_ROC[1:]-x_ROC[:-1]计算的是高(其实就是后一个x减去前一个x),y_ROC[:-1]+y_ROC[1:]是上底加下底(前一个y+后一个y）
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])
    # recall 就是tpr
    recall_list = tpr
    # 预测为正例中预测对的比例（阈值越大，pr也大）
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    # 保证pr曲线从（0,1）开始，1,0 结束
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    plt.plot(x_PR,y_PR)
    plt.show()
    # 计算无数个梯形面积来拟合pr曲线下的面积 x_PR[1:]-x_PR[:-1]计算的是高(其实就是后一个x减去前一个x),y_PR[:-1]+y_PR[1:]是上底加下底(集前一个y+后一个y）
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    # 分类正确的比率
    accuracy_list = (TP + TN) / len(real_score.T)
    # 被分对的负例占负例的比例
    specificity_list = TN / (TN + FP)
    #这里以是的f1-score最大时候所对应的阈值几算的性能指标作为最佳性能指标
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    print('*'*10)
    print( ' auc:{:.4f} ,aupr:{:.4f},f1_score:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, specificity:{:.4f}, precision:{:.4f}'.format(auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision))
    return [auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision]
# real_score,predict_score,temp_drug_dis
def cv_model_evaluate(interaction_matrix, predict_matrix, train_matrix):
    test_index = np.where(train_matrix == 0)
    real_score = interaction_matrix[test_index]
    predict_score = predict_matrix[test_index]
    return get_metrics(real_score, predict_score)
def get_metrics_topk(real_score, predict_score,topk):
    #这里得到每种药物评分最高的前topk个疾病对应的索引
    # index_array=np.argpartition(predict_score,kth=-1*topk,axis=-1)[:,-1*topk:]
    index_array=np.argpartition(predict_score.T,kth=-1*topk,axis=-1)[:,-1*topk:]
    # print(index_array)
    new_predict_score=np.zeros_like(real_score)
    for i in range(real_score.shape[1]):
        for j in index_array[i]:
            new_predict_score[j][i]=1.0
    # print(new_predict_score)
    #获得每种药物与疾病评分最高对应real_socre 的矩阵(这里认为每个药物与topk个疾病是有关联的，也就是预测为1)
    # topk_array=np.take_along_axis(real_score, index_array, axis=-1)
    topk_array=np.take_along_axis(real_score.T, index_array, axis=-1)
    # print(topk_array)
    #用预测的为正例的数量除以所有真实为整例的数量就是recall
    # print('recall:{:.4f}'.format(np.sum(topk_array)/np.sum(real_score)))
    # recall=get_metrics(real_score,topk_array)
    return np.sum(topk_array)/np.sum(real_score)
    # return get_metrics(real_score,new_predict_score)[4]
    # return recall_score(real_score.flatten(),new_predict_score.flatten())
if __name__ == '__main__':
    # predict_score=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.81,0.82,0.83,0.9]).reshape(3,4)
    # # print(predict_score)
    # real_score=np.array([0,0,0,0,1,0,1,0,0,1,1,1]).reshape(3,4)
    # get_metrics_topk(real_score, predict_score,2)
    # print(real_score)
    # get_metrics_topk(real_score, predict_score,2)
#     top_Ks=[]
#     K_value=[1,5,10,20,50,100,150,200]
#     real_score=np.loadtxt('../Dataset1/drug_dis.csv',delimiter=',')
#     for k in range(5):
#         top_K=[]
#         predict_score=np.loadtxt('../5-fold-dataset/VGAE-DR/VGAE-DR_predict_fold_{}.csv'.format(k))
#         # predict_score=sco.loadmat('../BNNR_predict_score{}.mat'.format(k+1))['M_recovery']
#         for top in K_value:
#             topk=get_metrics_topk(real_score,predict_score,top)
#             top_K.append(topk)
#         print(k,top_K)
#         top_Ks.append(top_K)
# # average_metries= np.array(result).sum(axis=0) / k_folds
#     average_top_k=np.array(top_Ks).sum(axis=0)/5
#     print(average_top_k)
    real=np.loadtxt('../Dataset1/drug_dis.csv',delimiter=',')
    real_index=np.where(real[217]==1)
    print(real_index)
    # predict=np.loadtxt('../Dataset1/case_study/new_drug_disease/VGAE-DR_all_drug_disease_predict_fold_0.csv')
    predict=np.loadtxt('../5-fold-dataset/VGAE-DR/VGAE-DR_all_drug_disease_predict_fold_0.csv')
    predict[217,real_index]=0
    print(sorted(predict[217]))
    # get_metrics(real,predict)
