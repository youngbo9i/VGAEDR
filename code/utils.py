import numpy as np
import torch


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



