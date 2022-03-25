import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self,in_dim,out_dim,drop=0.4,bias=False,activation=None):
        super(GraphConv,self).__init__()
        self.dropout = nn.Dropout(drop)
        self.activation = activation
        self.w = nn.Linear(in_dim,out_dim,bias=bias)
        nn.init.xavier_uniform_(self.w.weight)
        self.bias = bias
        if self.bias:
            nn.init.zeros_(self.w.bias)
    
    def forward(self,adj,x):
        x = self.dropout(x)
        x = adj.mm(x)
        x = self.w(x)
        if self.activation:
            return self.activation(x)
        else:
            return x

class AE(nn.Module):
    def __init__(self,feat_dim,hid_dim,out_dim,bias=False):
        super(AE,self).__init__()
        self.conv1 = GraphConv(feat_dim,hid_dim,bias=bias,activation=F.relu)
        #自己加的
        # self.conv5=GraphConv(hid_dim,hid_dim,bias=bias,activation=torch.sigmoid)
        self.mu = GraphConv(hid_dim,out_dim,bias=bias,activation=torch.sigmoid)
        self.conv3 = GraphConv(out_dim,hid_dim,bias=bias,activation=F.relu)
        self.conv4 = GraphConv(hid_dim,feat_dim,bias=bias,activation=torch.sigmoid)
        self.logvar = GraphConv(hid_dim,out_dim,bias=bias,activation=torch.sigmoid)

    def encoder(self,g,x):
        x = self.conv1(g,x)
        #我加的
        # x=self.conv5(g,x)
        h = self.mu(g,x)
        std = self.logvar(g,x)
        return h,std
    
    def decoder(self,g,x):
        x = self.conv3(g,x)
        x = self.conv4(g,x)
        return x
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            #这里实际上就是通过标准差和均值得到z
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self,g,x):
        mu,logvar = self.encoder(g,x)
        z = self.reparameterize(mu, logvar)
        return mu,logvar,self.decoder(g,z)

class LP(nn.Module):
    def __init__(self,hid_dim,out_dim,bias=False):
        super(LP,self).__init__()
        self.res1 = GraphConv(out_dim,hid_dim,bias=bias,activation=F.relu)
        # self.res2 = GraphConv(hid_dim,hid_dim,bias=bias,activation=torch.tanh)
        # self.res3 = GraphConv(hid_dim,hid_dim,bias=bias,activation=F.relu)
        self.res2 = GraphConv(hid_dim,out_dim,bias=bias,activation=torch.sigmoid)

    def forward(self,g,z):
        z = self.res1(g,z)
        res = self.res2(g,z)
        return res,z
    # def __init__(self, hid_dim, out_dim, bias=False):
    #     super(LP, self).__init__()
    #     self.res1 = GraphConv(out_dim, hid_dim, bias=bias, activation=F.relu)
    #     self.res2 = GraphConv(hid_dim,128,bias=bias,activation=torch.tanh)
    #     self.res3 = GraphConv(128,hid_dim,bias=bias,activation=F.relu)
    #     self.res4 = GraphConv(hid_dim, out_dim, bias=bias, activation=torch.sigmoid)
    # def forward(self,g,z):
    #     z = self.res2(g,self.res1(g,z))
    #     # print(z.size())
    #     res = self.res4(g,self.res3(g,z))
    #     return res,z

# class GNNq(nn.Module):
#     def __init__(self,drug_sim,dis_sim,args):
#         super(GNNq,self).__init__()
#         self.gnnq_drug = AE(drug_sim.shape[1], 256, args.hidden)
#         self.gnnq_dis = AE(dis_sim.shape[0], 256, args.hidden)
#
#     def forward(self, x_drug0, x_dis0,g_drug,g_dis):
#         h_drug,std_drug,x_drug = self.gnnq_drug(g_drug, x_drug0)
#         h_dis,std_dis,x_dis = self.gnnq_dis(g_dis, x_dis0)
#         return h_drug,std_drug,x_drug,h_dis,std_dis,x_dis
#
# class GNNp(nn.Module):
#     def __init__(self,drug_dis,args):
#         super(GNNp,self).__init__()
#         self.gnnp_drug = LP(args.hidden, drug_dis.shape[1])
#         self.gnnp_dis = LP(args.hidden, drug_dis.shape[0])
#
#     def forward(self,y0,g_drug,g_dis):
#         y_drug,z_drug = self.gnnp_drug(g_drug, y0)
#         y_dis,z_dis = self.gnnp_dis(g_dis, y0.t())
#         return y_drug,z_drug,y_dis,z_dis
