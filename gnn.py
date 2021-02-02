import match
import numpy as np
import json
from astropy.stats import bayesian_blocks
from astropy import units
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import minisom
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from tool import *
import copy

from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Data
from torch_scatter import scatter_mean
from torch.utils.data import Dataset, DataLoader


F_x=8
F_e=8
F_u=1

def Loss(cprob,OIII4960,classes,c_t=0,total_time=30):
    temp = OIII4960.view(-1,1).repeat(1,len(classes))
    snrs = torch.sqrt(1e-8+classes)*temp/3.*5.
    LeakyReLU = nn.LeakyReLU(0.)
    pergalutils = torch.sum(torch.sigmoid(5*(snrs-20))*cprob)
    num = torch.sum(cprob*torch.sigmoid(5*(snrs-5)))
    time_constraint = c_t*torch.sum(LeakyReLU(torch.sum(classes*cprob,dim=(-2,-1))-total_time))
    return -pergalutils+time_constraint,-pergalutils,num

class EdgeModel(torch.nn.Module):
    def __init__(self):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Seq(Lin(F_x*2+F_e+F_u,1*(F_x*2+F_e+F_u)), ReLU(), Lin(1*(F_x*2+F_e+F_u),F_e))

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u.view(1,F_u).repeat(len(src),1)], -1)
        out =  self.edge_mlp(out)
        return out
class NodeModel(torch.nn.Module):
    def __init__(self):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(F_e+F_x,1*(F_e+F_x)), ReLU(), Lin(1*(F_e+F_x),1*(F_e+F_x)))
        self.node_mlp_2 = Seq(Lin((1*F_e+2*F_x+F_u),1*(1*F_e+2*F_x+F_u)), ReLU(), Lin(1*(1*F_e+2*F_x+F_u),F_x))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u.view(1,F_u).repeat(len(x),1)], dim=1)
        out = self.node_mlp_2(out)
        return out
class GlobalModel(torch.nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.global_mlp = Seq(Lin(F_u+F_x,1*(F_u+F_x)), ReLU(), Lin(1*(F_u+F_x), F_u))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, torch.mean(x,dim=0)], dim=0)
        return self.global_mlp(out)


class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN,self).__init__()
        self.block1 = MetaLayer(EdgeModel(),NodeModel(),GlobalModel())
        self.block2 = MetaLayer(EdgeModel(),NodeModel(),GlobalModel())
        self.block3 = MetaLayer(EdgeModel(),NodeModel(),GlobalModel())
    def forward(self,data,batch=None):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        u = data.u
        x,edge_attr,u = self.block1(x,edge_index,edge_attr,u,batch)
        x,edge_attr,u = self.block2(x,edge_index,edge_attr,u,batch)
        x,edge_attr,u = self.block3(x,edge_index,edge_attr,u,batch)
        return x

def Region_to_Graph(region,maxtime=30):
    region = np.array(region)
    prob = np.zeros((7,len(region)))
    prob[0,:]+=1
    vertices = np.vstack([region,prob]).T
    edge_attr = []
    edge_index = []
    for i in range(len(region)):
        for j in range(len(region)-i):
            edge_index.append([i,i+j])
            edge_attr.append(vertices[i]-vertices[i+j])
        for j in range(i):
            edge_index.append([i,j])
            edge_attr.append(vertices[i]-vertices[j])
    edge_attr = torch.tensor(edge_attr).float()
    edge_index = torch.tensor(edge_index,dtype=torch.long)
    u=maxtime*torch.ones(F_u).float() #default value for global parameters
    vertices = torch.tensor(vertices,dtype=torch.float)
    data = Data(x=vertices.cuda(),edge_index=edge_index.t().contiguous().cuda(),edge_attr=edge_attr.cuda(),u=u.cuda())
    return data

class Loader(Dataset):
    def __init__(self,regions):
        self.regions = regions
    def __len__(self):
        return len(self.regions)
    def __getitem__(self,idx):
        graph = Region_to_Graph(self.regions[idx])
        return graph



if __name__ == '__main__':
    nepoch = 20
    batchsize = 1
    nround = 15
    lr_t = 5e-4
    lr_p = 1e-4
    train_t = True

    ### Load Whole Dataset ###
    SM,data,ra,dec,OIII4960,redshift=LoadUniverseMachine(0)

    field_center_x = -0.5
    field_center_y = -0.5
    # Transfer to Focal Plane
    target_list = SkyToFocalPlane(ra,dec,field_center_x,field_center_y)
    # Constrained to Telescope Field
    target_list,SM,data,OIII4960,ra,dec,redshift = ReduceSize(target_list,SM,data,OIII4960,ra,dec,redshift,maxnum=10000000)
    # Load SOM
    som,utils_weights,z_weights = LoadSOM()
    # Prepare for Allocation
    som_pos = np.array([som.winner(idata) for idata in data])

    def wrap_trimatch(pos):
        return tuple(match.trimatch(pos[0],pos[1]))
    indices = tuple(map(wrap_trimatch,target_list))

    # Fibers 
    all_data = [[] for i in range(2394)]
    for i in range(len(indices)):
        if OIII4960[i]>1.5:
            for j in indices[i]:
                if j<2394:
                    all_data[j].append(OIII4960[i])
    dataset = Loader(all_data)

    
    ### Train GNN ###
    gnn = GNN().cuda()
    optimizer = optim.Adam(gnn.parameters(),lr=lr_t)

    N = 5
    c_ts = 5**np.linspace(-1,-0.5,N)

    classes = torch.tensor([0.,2.,4.,6.,8.,10.,12.]).cuda()
    softmax = nn.Softmax(dim=-1)
    for ntry in range(N):
        try:
            gnn.load_state_dict(torch.load('model_gnn.pth'))
            gnn.eval()
        except:
            print('No available GNN model exists')
        minloss=1e9
        c_t = c_ts[ntry]
        print(c_t)
        for i_epoch in range(nepoch):
            loss = 0
            utils = 0
            num = 0
            maxtime = 30
            for i in range(max(dataset.__len__(),100)):
                gnn.zero_grad()
                sample = dataset.__getitem__(i)
                x = gnn(sample)
                cprob = softmax(x[:,1:])
                line = sample.x[:,0]
                iloss,iutils,inum =  Loss(cprob,line,classes,c_t=c_t,total_time=sample.u[0])
                num += inum
                loss += iloss
                utils += iutils
            nonseputil = 40000*torch.sigmoid(num-20000)
            loss -= nonseputil
            utils -= nonseputil
            loss.backward()
            if train_t:
                optimizer.step()
            print(utils.data,loss.data,num.data)
            if minloss>loss:
                minloss=loss
                if train_t:
                    torch.save(gnn.state_dict(), 'model_gnn.pth')

    gnn.load_state_dict(torch.load('model_gnn.pth'))
    gnn.eval()
    x = dataset.__getitem__(101)
    y = gnn(x)
    x = y[:,0].cpu().detach().numpy().flatten()
    y = y[:,1:].cpu().detach().numpy()
    fig,ax = plt.subplots()
    ax.plot(x,y) #dist = ax.imshow(y.T,aspect='auto',extent=[0,20,0,14],origin='lower',interpolation='nearest')
    ax.set_xlabel('OIII4960')
    ax.set_ylabel('Time')
    #cbar = fig.colorbar(dist,ax=ax)
    #cbar.minorticks_on()
    plt.savefig('time_arrange.png')
    #plt.show()
