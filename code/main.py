from random import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from models import GraphConv, AE, LP
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Dimension of representations')
#0.7
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Weight between drug space and disease space')
parser.add_argument('--data', type=int, default=1, choices=[1,2],
                    help='Dataset')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
set_seed(args.seed, args.cuda)
sim1_path = '../Dataset1/dis_sim.csv'
sim2_path = '../Dataset1/drug_sim.csv'
association_path = '../Dataset1/drug_dis.csv'
Therapeutic_association_path = '../Dataset1/Therapeutic_association.txt'
if args.data==1:
    dis_sim, drug_dis, drug_sim, g_drug, g_dis = load_data(sim1_path, sim2_path, association_path, args.cuda)
if args.data==2:
    dis_sim, drug_dis, drug_sim, g_drug, g_dis = load_data(sim1_path, sim2_path, Therapeutic_association_path, args.cuda)
class GNNq(nn.Module):
    def __init__(self):
        super(GNNq, self).__init__()
        self.gnnq_drug = AE (drug_sim.shape[1], 256, args.hidden)
        self.gnnq_dis = AE(dis_sim.shape[0], 256, args.hidden)

    def forward(self, x_drug0, x_dis0):
        h_drug, std_drug, x_drug = self.gnnq_drug(g_drug, x_drug0)
        h_dis, std_dis, x_dis = self.gnnq_dis(g_dis, x_dis0)
        return h_drug, std_drug, x_drug, h_dis, std_dis, x_dis

class GNNp(nn.Module):
    def __init__(self):
        super(GNNp, self).__init__()
        self.gnnp_drug = LP(args.hidden, drug_dis.shape[1])
        self.gnnp_dis = LP(args.hidden, drug_dis.shape[0])

    def forward(self, y0):
        y_drug, z_drug = self.gnnp_drug(g_drug, y0)
        y_dis, z_dis = self.gnnp_dis(g_dis, y0.t())
        return y_drug, z_drug, y_dis, z_dis

def criterion(output, target, msg, n_nodes, mu, logvar):
    cost = F.binary_cross_entropy(output, target)
    KL = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KL


def train(gnnq, gnnp, x_drug0, x_dis0, y0, epoch, alpha):
    beta0 = 0.1
    gamma0 = 0.1
    optp = torch.optim.Adam(gnnp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optq = torch.optim.Adam(gnnq.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for e in range(epoch):
        gnnq.train()
        h_drug, std_drug, x_drug, h_dis, std_dis, x_dis = gnnq(x_drug0, x_dis0)
        lossql = criterion(x_drug, x_drug0, "drug", g_drug.shape[0], h_drug, std_drug)
        lossqd = criterion(x_dis, x_dis0, "disease", g_dis.shape[0], h_dis, std_dis)
        lossq = alpha * lossql + (1 - alpha) * lossqd + beta0  * F.mse_loss(torch.mm(h_drug, h_dis.t()), y0)
        optq.zero_grad()
        lossq.backward()
        optq.step()
        gnnq.eval()
        with torch.no_grad():
            h_drug, _, _, h_dis, _, _ = gnnq(x_drug0, x_dis0)
        gnnp.train()
        y_drug, z_drug, y_dis, z_dis = gnnp(y0)
        losspl = F.binary_cross_entropy(y_drug, y0) + gamma0  * F.mse_loss(z_drug, h_drug)
        losspd = F.binary_cross_entropy(y_dis, y0.t()) + gamma0  * F.mse_loss(z_dis, h_dis)
        lossp = alpha * losspl + (1 - alpha) * losspd
        optp.zero_grad()
        lossp.backward()
        optp.step()
        gnnp.eval()
        with torch.no_grad():
            y_drug, _, y_dis, _ = gnnp(y0)
        if e % 20 == 0:
            print('Epoch %d | Lossp: %.4f | Lossq: %.4f' % (e, lossp.item(), lossq.item()))
    return alpha * y_drug + (1 - alpha) * y_dis.t()


def cross_validation_experiment(k_folds, drug_dis, alpha=0.5):
    print("Dataset{}, {}-fold CV".format(args.data, k_folds))
    result = []
    drug_num = drug_dis.shape[0]
    drug_idx = np.arange(drug_num)
    np.random.shuffle(drug_idx)
    for k in range(k_folds):
        print("Fold {}".format(k + 1))
        temp_drug_dis = drug_dis.clone()
        print('before:', torch.sum(temp_drug_dis.reshape(-1)))
        for j in range(k * drug_num // k_folds, (k + 1) * drug_num // k_folds):
            temp_drug_dis[drug_idx[j], :] = 0
        gnnq = GNNq()
        gnnp = GNNp()
        if args.cuda:
            gnnq = gnnq.cuda()
            gnnp = gnnp.cuda()
        print('after:', torch.sum(temp_drug_dis.reshape(-1)))
        train(gnnq, gnnp, drug_sim, dis_sim.t(), temp_drug_dis, args.epochs, args.alpha)
        gnnq.eval()
        gnnp.eval()
        yli, _, ydi, _ = gnnp(temp_drug_dis)
        resi = alpha * yli + (1 - alpha) * ydi.t()
        resi = scaley(resi)
        if args.cuda:
            real_score, predict_score = drug_dis.cpu().detach().numpy(), resi.cpu().detach().numpy()
            temp_drug_dis = temp_drug_dis.cpu().detach().numpy()
        else:
            real_score, predict_score = drug_dis.detach().numpy(), resi.detach().numpy()
            temp_drug_dis = temp_drug_dis.detach().numpy()
        metrics = get_metrics(real_score, predict_score)
        result.append(metrics)
    print('--5 flod cv average result----')
    return result


if __name__ == '__main__':
    title = 'result--dataset' + str(args.data)
    k_folds = 5
    result = cross_validation_experiment(k_folds, drug_dis, alpha=args.alpha)
    average_metries = np.array(result).sum(axis=0) / k_folds
    print(average_metries[0], average_metries[1], average_metries[2], average_metries[3], average_metries[4],
          average_metries[5], average_metries[6])
