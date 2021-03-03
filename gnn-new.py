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

from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Data,DataLoader
from torch_scatter import scatter_mean,scatter
from torch.utils.data import Dataset
from torch.autograd import Variable

F_x=4
F_u=1

F_e_in=4 # input edge attr num
F_e=2 # output edge attr num

def conv(cprob):
    a = cprob[:,0,:]
    b = cprob[:,1,:]
    result = b.clone()*a[:,0].view(-1,1)
    for i in range(F_e-1):
        result[:,i+1:]+= b[:,:-(1+i)]*a[:,i+1].view(-1,1)
    result[:,-1]=1-torch.sum(result[:,:-1],dim=-1)
    return result

def gumbel(cprob,classes):
    tau = 1 
    hard = False
    time,class_arg = torch.max(F.gumbel_softmax(torch.log(cprob+1e-8),tau=tau,hard=hard)*classes,dim=-1)
    #time = torch.sum(F.gumbel_softmax(torch.log(cprob+1e-8),tau=tau,hard=hard)*classes,dim=-1)
    return time

def Loss(cprob,graph,classes,c_t,total_time=30,damp=0,use_gumbel=False):
    edge_index = graph.edge_index
    reward = graph.edge_attr[:,:-1]
    reward_set = torch.mean(reward.view(-1,2,3),dim=1)
    reward_set = torch.sum(reward_set,dim=-1)
    weight = graph.edge_attr[:,-1]
 
    time = gumbel(cprob,classes)*weight
    time_sum = torch.sum(time.view(-1,2),dim=-1)
    time_sum = time_sum-F.relu(time_sum-2) #no more than 2hr per galaxy
    
    obs = 1-torch.exp(-5.*time_sum)
    totutils = torch.sum(obs*reward_set)

    spent_time = torch.zeros(2394).float().cuda()
    used_time = scatter(time,edge_index[0],reduce='sum')
    spent_time[:len(used_time)]=used_time

    LeakyReLU = nn.LeakyReLU(1e-3)
    delta = LeakyReLU(spent_time-total_time)
    time_constraint = torch.sum((c_t+damp*delta.detach())*delta)
        

    return -totutils+time_constraint,totutils,delta

class EdgeModel_in(torch.nn.Module):
    def __init__(self):
        super(EdgeModel_in, self).__init__()
        self.edge_mlp = Seq(Lin(F_x*2+F_e_in+F_u,1*(F_x*2+F_e_in+F_u)), LeakyReLU(0.1), Lin(1*(F_x*2+F_e_in+F_u),F_e))

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out =  self.edge_mlp(out)
        return out
class EdgeModel(torch.nn.Module):
    def __init__(self):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Seq(Lin(F_x*2+F_e+F_u,1*(F_x*2+F_e+F_u)), LeakyReLU(0.1), Lin(1*(F_x*2+F_e+F_u),F_e))

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out =  self.edge_mlp(out)
        return out
class NodeModel(torch.nn.Module):
    def __init__(self):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(F_e+F_x,1*(F_e+F_x)), LeakyReLU(0.1), Lin(1*(F_e+F_x),1*(F_e+F_x)))
        self.node_mlp_2 = Seq(Lin((2*F_e+3*F_x+F_u),1*(2*F_e+3*F_x+F_u)), LeakyReLU(0.1), Lin(1*(2*F_e+3*F_x+F_u),F_x))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row],edge_attr], dim=1)
        a = scatter(out, col, dim=0, dim_size=x.size(0),reduce='sum')/100.0
        b = scatter(out, col, dim=0, dim_size=x.size(0),reduce='mean')
        out = torch.cat([x, a, b, u[batch]], dim=1)
        out = self.node_mlp_2(out)
        return out
class GlobalModel(torch.nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.global_mlp = Seq(Lin(F_u+F_x,1*(F_u+F_x)), LeakyReLU(0.1), Lin(1*(F_u+F_x), F_u))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter_mean(x,batch,dim=0)], dim=1)
        return self.global_mlp(out)


class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN,self).__init__()
        self.block1 = MetaLayer(EdgeModel_in(),NodeModel(),GlobalModel())
        self.block2 = MetaLayer(EdgeModel(),NodeModel(),GlobalModel())
        self.block3 = MetaLayer(EdgeModel(),NodeModel(),GlobalModel())
    def forward(self,data,batch=None,maxtime=30):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        u = data.u
        row,col = edge_index
        x,edge_attr,u = self.block1(x,edge_index,edge_attr,u,batch)
        x,edge_attr,u = self.block2(x,edge_index,edge_attr,u,batch)
        x,edge_attr,u = self.block3(x,edge_index,edge_attr,u,batch)
        softmax = nn.Softmax(dim=-1)
        return softmax(edge_attr),edge_index

def Region_to_Graph(indices,properties,args,maxtime=30):
    properties = np.array(properties)
    edge_attr = []
    edge_index = []
    for i,index in enumerate(indices):
        if args[i]:
            a=-1
            b=-1
            for j in range(len(index)):
                if index[j]<2394:
                    if a==-1:
                        a=index[j]
                    else:
                        b=index[j]
            if b==-1:
                edge_attr.append(np.append(properties[i],0.5))
                edge_index.append([a,a])
                edge_attr.append(np.append(properties[i],0.5))
                edge_index.append([a,a])
            else:
                edge_attr.append(np.append(properties[i],1.0))
                edge_attr.append(np.append(properties[i],1.0))
                edge_index.append([a,b])
                edge_index.append([b,a])
            
            
    edge_attr = torch.tensor(edge_attr).float()
    edge_index = torch.tensor(edge_index,dtype=torch.long)
    vertices = torch.zeros(2394,F_x).float()  #default value for vertices parameters, containing total time
    u=torch.tensor([np.zeros(F_u)]).float() #default value for global parameters
    data = Data(x=vertices.cuda(),edge_index=edge_index.t().contiguous().cuda(),edge_attr=edge_attr.cuda(),u=u.cuda())
    return data

class Loader(Dataset):
    def __init__(self,indices_list,properties_list,args_list):
        self.indices_list = indices_list
        self.properties_list = properties_list
        self.args_list = args_list
    def __len__(self):
        return len(self.indices_list)
    def __getitem__(self,idx):
        graph = Region_to_Graph(self.indices_list[idx],self.properties_list[idx],self.args_list[idx])
        return graph


if __name__ == '__main__':
    nepoch = 200
    batchsize = 1
    nround = 15
    lr_t = 1e-4
    lr_c = 0#1e-7
    damp = 1e-6

    # Load Whole Dataset
    ra = []
    dec = []
    utils = []
    for i in range(16):
        ra_i,dec_i,utils_i=LoadUniverseMachine(13)
        xsize = max(ra_i)-min(ra_i)
        ysize = max(dec_i)-min(dec_i)
        ra.append(ra_i+i//4*xsize)
        dec.append(dec_i+i%4*ysize)
        #print(min(ra[-1]),max(ra[-1]),min(dec[-1]),max(dec[-1]))
        utils.append(utils_i)
    ra = np.concatenate(ra).flatten()
    dec = np.concatenate(dec).flatten()
    utils = np.concatenate(utils).reshape(-1,3)

    field_center_x = 0.5*(max(ra)+min(ra))
    field_center_y = 0.5*(max(dec)+min(dec))
    # Transfer to Focal Plane
    target_list = SkyToFocalPlane(ra,dec,field_center_x,field_center_y)
    # Constrained to Telescope Field
    target_list,ra,dec,utils = ReduceSize(target_list,ra,dec,utils,maxnum=10000000)
    
    
    def wrap_trimatch(pos):
        return tuple(match.trimatch(pos[0],pos[1]))
    indices = tuple(map(wrap_trimatch,target_list))

    args = ra>-100 # All True
    for i in range(len(indices)):
        index = indices[i]
        num = 0
        for j in range(len(index)):
            if index[j]<2394:
                num+=1
        if num==0 or num==3: #No fiber or 3 fibers
            args[i] = False

    dataset = Loader([indices],[utils],[args])
    dataloader = DataLoader(dataset,batch_size=1)


    gnn = GNN().cuda()
    optimizer = optim.Adam(gnn.parameters(),lr=lr_t)


    softmax = nn.Softmax(dim=-1)
    classes = torch.tensor([0.,2.]).float().cuda()

    try:
        gnn.load_state_dict(torch.load('model_gnn4.pth'))
        gnn.eval()
    except:
        print('No available GNN model exists')
    minloss=1e9

    try:
        c_t = torch.load('multiplier_gnn4.pth')
    except:
        print('No multiplier saved')
        c_t = torch.nn.Parameter(0.0*torch.ones(2394).float().cuda(),requires_grad=True)
    for i_epoch in range(nepoch):
        gnn.zero_grad()
        for i_batch,graph in enumerate(dataloader):
            edge_attr,edge_index = gnn(graph,graph.batch)
            loss,utils,overtime = Loss(edge_attr,graph,classes,c_t,total_time=30,damp=damp)
        print(loss.item(),utils.item(),torch.max(overtime).item(),len(overtime[overtime>0]))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            c_t +=  lr_c*c_t.grad
            c_t = F.relu(c_t).requires_grad_(True)
        if c_t.grad is not None:
            c_t.grad.zero_()
    torch.save(gnn.state_dict(), 'model_gnn4.pth')
    torch.save(c_t,'multiplier_gnn4.pth')
    print(torch.mean(c_t),torch.max(c_t))

