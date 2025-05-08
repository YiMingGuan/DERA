
import numpy as np
import dgl
import torch
from bidict import bidict
from copy import deepcopy
import os
# from simcse import SimCSE
import ot

class KG:
    def __init__(self):
        self.rel_ids = bidict()
        self.ent_ids = bidict()
        self.er_e = dict()
        self.ee_r = dict()
        self.edges = set()

    def construct_graph(self):
        ent_graph = dgl.graph(list(self.edges))
        self.ent_graph = dgl.to_bidirected(ent_graph).to_simple()



class EAData:
    def __init__(self,kgs, bi=True):
        self.kg = [KG(), KG()]
        self.convert_kgs(kgs, kg_id=0,bi=bi)
        self.convert_kgs(kgs, kg_id=1,bi=bi)
        self.seed_pair = bidict()
        self.test_pair = bidict()
        self.convert_pairs(kgs)
        self.final_test_pair = deepcopy(self.test_pair)
        self.test_pair.update(self.seed_pair)
        self.seed_pair = bidict()
        self.name = kgs.kg1.kg_name
        """
        if os.path.exists(self.loc+'hard_pairs.txt'):
            self.hard_pair = {}
            with open(self.loc+'hard_pairs.txt', 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    head, tail = line.strip().split('\t')
                    self.hard_pair[int(head)] = int(tail)
        """
    
    def convert_pairs(self, kgs):
        for (ent1_id, ent2_id) in kgs.train_pairs.items():
            self.seed_pair[ent1_id] = ent2_id
        
        for (ent1_id, ent2_id) in kgs.test_pairs.items():
            self.test_pair[ent1_id] = ent2_id

        
    
    def convert_kgs(self, kgs, kg_id, bi=True):
        kg = getattr(kgs, f"kg{kg_id + 1}")
        """
        ent_ids
        """
        for ent_id in kg.ent_ids:
            self.kg[kg_id].ent_ids[ent_id] = kg.ent_ids[ent_id]
        
        for rel_id in kg.rel_ids:
            self.kg[kg_id].rel_ids[rel_id] = kg.rel_ids[rel_id]
        """
        rel_triples
        """
        for ent in kg.entities.values():
            for rel in ent.relationships:
                if rel.is_inv:
                    continue
                for tail in ent.relationships[rel]:
                    head_id = self.kg[kg_id].ent_ids.inv[ent.url]
                    rel_id = self.kg[kg_id].rel_ids.inv[rel.url]
                    tail_id = self.kg[kg_id].ent_ids.inv[tail.url]
                    self.kg[kg_id].edges.add((head_id, tail_id))
                    self.kg[kg_id].er_e[(head_id, rel_id)] = tail_id
                    self.kg[kg_id].ee_r[(head_id, tail_id)] = rel_id
                    if bi:
                        self.kg[kg_id].er_e[(tail_id, rel_id)] = head_id
        self.kg[kg_id].construct_graph()



def NeuralSinkhorn(cost, p_s=None, p_t=None, trans=None, beta=0.1, outer_iter=20):
    if p_s is None:
        p_s = torch.ones([cost.shape[0],1],device=cost.device)/cost.shape[0]
    if p_t is None:
        p_t = torch.ones([cost.shape[1],1],device=cost.device)/cost.shape[1]
    if trans is None:
        trans = p_s @ p_t.T
    a = torch.ones([cost.shape[0],1],device=cost.device)/cost.shape[0]
    cost_new = torch.exp(-cost / beta)
    for oi in range(outer_iter):
        kernel = cost_new * trans
        b = p_t / (kernel.T@a)
        a = p_s / (kernel@b)
        trans = (a @ b.T) * kernel
    return trans

def VallinaSinkhorn(C, mu=None, nu=None, tol=1e-9, reg=0.1, num_iters=100):
    device = C.device
    n, m = C.shape
    if mu is None:
        mu = torch.ones(n, device=device) / n
    if nu is None:
        nu = torch.ones(m, device=device) / m
    
    K = torch.exp(-C / reg)
    u = torch.ones(n, device=device) / n
    v = torch.ones(m, device=device) / m
    
    for _ in range(num_iters):
        u_prev = u
        u = mu / (K @ v)
        v = nu / (K.t() @ u)
        # Convergence check
        err_u = torch.max(torch.abs(mu - u * (K @ v)))
        err_v = torch.max(torch.abs(nu - v * (K.t() @ u)))
        if err_u < tol and err_v < tol:
            break
    
    P = torch.outer(u, v) * K
    return P


def POT_Sinkhorn(C, mu=None, nu=None, tol=1e-9, reg=0.1, method='sinkhorn', num_iters=1000):
    device = C.device
    n, m = C.shape
    if mu is None:
        mu = torch.ones(n, device=device) / n
    if nu is None:
        nu = torch.ones(m, device=device) / m
    P = ot.sinkhorn(mu, nu, C, reg, stopThr=tol, numItermax=num_iters, method=method)
    return P

def test_align(pred, test_pair):
    ind = (-pred).argsort(axis=1)
    ind = ind.cpu().numpy()
    a1, a10, mrr = 0, 0, 0
    for k, v in test_pair.items():
        rank=np.where(ind[k]==v)[0][0]+1
        if rank==1:
            a1+=1
        if rank<=10:
            a10+=1
        mrr+=1/rank
    a1 /= len(test_pair)
    a10 /= len(test_pair)
    mrr /= len(test_pair)
    print('H@1 %.1f%% H@10 %.1f%% MRR %.1f%%' % (a1*100, a10*100, mrr*100))
    return a1, a10, mrr

def get_original_ranks(pred, test_pair, ent_ids1, ent_ids2):
    ind = (-pred).argsort(axis=1)
    ind = ind.cpu().numpy()
    ranks = {}
    for k, v in test_pair.items():
        q_ori_id = ent_ids1[k]
        rank = ind[k]
        ranks[q_ori_id] = [ent_ids2[i] for i in rank]
    return ranks

class GWEA():
    def __init__(self, data, args, ent_emb=None, kg_abs_path=None, trained=False):
        self.iters = 0
        self.trained = trained
        self.data = data
        self.candi = self.data.test_pair.copy()
        self.graph1 = self.data.kg[0].ent_graph
        self.graph2 = self.data.kg[1].ent_graph
        self.rel_list = [list(self.data.kg[0].rel_ids),list(self.data.kg[1].rel_ids)]
        self.ent_ids1 = bidict()
        self.ent_ids2 = bidict()
        self.args = args
        self.ent_emb = ent_emb

        if 'dw15kv2' not in self.data.name:
            self.rel_emb = np.load(os.path.join(kg_abs_path, 'rel_emb.npy'))

        self.ent_emb = self.ent_emb/((self.ent_emb**2).sum(1)**0.5)[:,None] 
        # self.rel_emb = self.rel_emb/((self.rel_emb**2).sum(1)**0.5)[:,None]
        # todo normalize embedding in advance
        rand_ind = np.random.permutation(len(data.test_pair))
        self.test_pair = {}
        for i, ind in enumerate(rand_ind):
            self.test_pair[i] = ind
        for i, (ent1, ent2) in enumerate(self.candi.items()):
            self.ent_ids1[i] = ent1
            self.ent_ids2[self.test_pair[i]] = ent2
        
        self.original_anchor = bidict()
        for i, (ent1, ent2) in enumerate(self.data.final_test_pair.items()):
            self.original_anchor[ent1] = ent2
        
        self.n = len(self.ent_ids1)
        self.ent_ids2 = bidict(sorted(self.ent_ids2.items()))
        # self.cost_s = self.graph1.subgraph(list(self.ent_ids1.values())).adj().cuda()
        # self.cost_t = self.graph2.subgraph(list(self.ent_ids2.values())).adj().cuda()
        self.cost_st_feat = 1-self.ent_emb[list(self.ent_ids1.values())]@self.ent_emb[list(self.ent_ids2.values())].T

    def cal_cost_st(self, w_homo=1, w_rel=1, w_feat=1, M=20):
        self.cost_st = torch.zeros(self.n, self.n)
        if w_homo>0:
            cost_st_homo = self.cal_cost_st_homo()
            cost_st_homo = cost_st_homo#+cost_st_homo.T)
            cost_st_homo[cost_st_homo>M]=M
            cost_st_homo = 1-cost_st_homo/cost_st_homo.max()
            self.cost_st += w_homo*cost_st_homo
        if w_rel>0:
            cost_st_rel = self.cal_cost_st_rel(bi=True)
            cost_st_rel = cost_st_rel#+cost_st_rel.T
            cost_st_rel[cost_st_rel>M]=M
            cost_st_rel = 1-cost_st_rel/cost_st_rel.max()
            self.cost_st += w_rel*cost_st_rel
        if w_feat>0:
            self.cost_st += w_feat*self.cost_st_feat
        self.cost_st = self.cost_st.cuda()

    def cal_cost_st_homo(self):
        cost = torch.zeros(self.n,self.n)
        for i, (ent1, ent2) in enumerate(self.anchor.items()):
            idx1, idx2 = [],[]
            for ne1 in self.graph1.predecessors(ent1).numpy():
                if ne1 in self.ent_ids1.values():
                    idx1.append(self.ent_ids1.inv[ne1])
            for ne2 in self.graph2.predecessors(ent2).numpy():
                if ne2 in self.ent_ids2.values():
                    idx2.append(self.ent_ids2.inv[ne2])
            if len(idx1)>0 and len(idx2) > 0:
                idxx = np.ix_(idx1,idx2)
                cost[idxx] += 1
        return cost

    def cal_cost_st_rel(self, bi=True):
        cost = torch.zeros(self.n,self.n)
        for (head, rel), tail in self.data.kg[0].er_e.items():
            if head in self.anchor.keys() and rel in self.r2r.keys() and tail in self.ent_ids1.values():
                head2 = self.anchor[head]
                rel2 = self.r2r[rel]
                if head2 in self.anchor.values() and (head2, rel2) in self.data.kg[1].er_e.keys():
                    tail2 = self.data.kg[1].er_e[(head2, rel2)]
                    if tail2 in self.ent_ids2.values():
                        cost[self.ent_ids1.inv[tail]][self.ent_ids2.inv[tail2]] += 1
        if bi:
            for (head, rel), tail in self.data.kg[1].er_e.items():
                if head in self.anchor.values() and rel in self.r2r.values() and tail in self.ent_ids2.values():
                    head2 = self.anchor.inv[head]
                    rel2 = self.r2r[rel]
                    if head2 in self.anchor.keys() and (head2, rel2) in self.data.kg[0].er_e.keys():
                        tail2 = self.data.kg[0].er_e[(head2, rel2)]
                        if tail2 in self.ent_ids1.values():
                            cost[self.ent_ids1.inv[tail2]][self.ent_ids2.inv[tail]] += 1
        return cost

    def update_anchor(self, X, thre=None):
        if thre is None:
            thre = 0.5/self.n
        val, idx = X.cpu().topk(1)
        x_max = X.cpu().max()
        anchor = bidict()
        knt, total, pre, rec, f1 = 0, 0, 0, 0, 0
        for i in range(len(idx)):
            if val[i] > x_max-thre:
                if self.ent_ids1[i] not in anchor.keys() and self.ent_ids2[idx[i][0].item()] not in anchor.values():
                    anchor[self.ent_ids1[i]] = self.ent_ids2[idx[i][0].item()]
                    total += 1
                    if idx[i][0].item() == self.test_pair[i]:
                        knt += 1
        rec = knt/len(self.test_pair)
        if total > 0:
            pre = knt/total
            f1 = (2*pre*rec)/(pre+rec)
        print(knt, total, len(self.test_pair), "thre:{:.2e}, pre: {:.4f}, rec: {:.4f}, f1: {:.4f}".format(thre,pre,rec,f1))
        
        if self.trained:
            """
            remove false positive
            """
            for i, (ent1, ent2) in enumerate(self.original_anchor.items()):
                if ent1 in anchor:
                    anchor.pop(ent1)
                if ent2 in anchor.inv:
                    anchor.inv.pop(ent2)
            
            """
            add true positive
            """
            for i, (ent1, ent2) in enumerate(self.original_anchor.items()):
                anchor[ent1] = ent2
        
        self.anchor = anchor
        return pre, rec, f1

    def rel_align(self, emb_w=1, seed_w=1, M=20):
        # (1) name channel
        rel_n1 = len(self.rel_list[0])
        rel_sim = torch.zeros(len(self.rel_list[0]),len(self.rel_list[1]))
        if emb_w > 0 and 'dw15kv2' not in self.data.name:
            rel_emb = torch.tensor(self.rel_emb)
            emb_rel_sim = rel_emb[self.rel_list[0]]@rel_emb[self.rel_list[1]].T
            emb_rel_sim = 1-emb_rel_sim.float()
            rel_sim += emb_w*emb_rel_sim
        # (2) structure channel
        if seed_w > 0:
            anchor_rel_sim = torch.zeros_like(rel_sim)
            for (head, rel), tail in self.data.kg[0].er_e.items():
                if head in self.anchor.keys() and tail in self.anchor.keys():
                    head2 = self.anchor[head]
                    tail2 = self.anchor[tail]
                    if head2 in self.anchor.values() and (head2, tail2) in self.data.kg[1].ee_r.keys():
                        rel2 = self.data.kg[1].ee_r[(head2, tail2)]
                        anchor_rel_sim[rel][rel2-rel_n1] += 1
            print("anchor_rel_mat:", anchor_rel_sim.sum())
            anchor_rel_sim[anchor_rel_sim>M]=M
            anchor_rel_sim = 1- anchor_rel_sim/anchor_rel_sim.max()
            rel_sim += seed_w*anchor_rel_sim

        rel_mat = NeuralSinkhorn(rel_sim)
        self.r2r = {}
        for idx1, idx2 in enumerate(list(rel_mat.argmax(1).numpy())):
            self.r2r[idx1] = rel_n1 + idx2
        for idx2, idx1 in enumerate(list(rel_mat.argmax(0).numpy())):
            self.r2r[rel_n1 + idx2] = idx1

    def ot_align(self, initX=None, beta=0.1, iter=10):
        print("===original emb similarity align result===")
        test_align(1.0 - self.cost_st, self.test_pair)

        trans = NeuralSinkhorn(self.cost_st, beta=beta, trans=initX, outer_iter=iter)
        print("===OT align result===")
        test_align(trans, self.test_pair)
        """
        vallina_trans = VallinaSinkhorn(self.cost_st, reg=beta, num_iters=iter)
        print("===Vallina align result===")
        test_align(vallina_trans, self.test_pair)

        methods = ['sinkhorn', 'sinkhorn_log', 'sinkhorn_stabilized', 'sinkhorn_epsilon_scaling']
        for method in methods:
            pot_trans = POT_Sinkhorn(self.cost_st, reg=0.05, method=method)
            print("===POT align result: {}===".format(method))
            test_align(pot_trans, self.test_pair)
        """
        return trans
    """
    def gw_align(self, initX=None, lr=0.001, iter=200, alpha=1000):
        alpha = 2*self.n*self.n/(self.cost_s.to_dense().sum()+self.cost_t.to_dense().sum()).cpu().item()
        trans = self.gw_torch(self.cost_s, self.cost_t, alpha, trans=initX, beta=lr, outer_iter=iter, test_pair=self.test_pair)
        print("===GW align result===")
        test_align(trans, self.test_pair)
        return trans

    def gw_torch(self, cost_s, cost_t, alpha=None, p_s=None, p_t=None, trans=None, beta=0.001,
                outer_iter=1000, inner_iter=10, test_pair=None):
        print("=======size======")
        print("cost_s: ", cost_s.to_dense().size())
        print("cost_t: ", cost_t.to_dense().size())


        device = cost_s.device
        last_fgw_score = 100
        knt = 0
        if p_s is None:
            p_s = torch.ones([cost_s.shape[0],1], device=device)/cost_s.shape[0]
        if p_t is None:
            p_t = torch.ones([cost_t.shape[0],1], device=device)/cost_t.shape[0]
        if trans is None:
            trans = p_s @ p_t.T
        print("trans: ", trans.size())
        for oi in range(outer_iter):
            cost = - 2 * cost_t @ (cost_s @ trans).T
            cost = cost.T  
            kernel = torch.exp(-cost / beta) * trans
            a = torch.ones_like(p_s)/p_s.shape[0]
            for ii in range(inner_iter):
                b = p_t / (kernel.T@a)
                a_new = p_s / (kernel@b)
                a = a_new
            trans = (a @ b.T) * kernel
            if oi % 20 == 0:
                test_align(trans, test_pair)
                gw_score = -torch.trace(cost_s.to_dense() @ trans @ cost_t.to_dense() @ trans.T).cpu().item()
                ot_score = (self.cost_st*trans).sum().cpu().item()
                fgw_score = alpha*gw_score + ot_score
                print(gw_score, ot_score, fgw_score)
                self.iters = oi
                if fgw_score - last_fgw_score > -0.00002:
                    knt += 1
                    if knt >= 2:
                        break
                last_fgw_score = fgw_score
        return trans
    
    def embedding_match():
        pass
    """