import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.base_MAGNN import MAGNN_ctr_ntype_specific
from model.HGNN_AC import HGNN_AC
from utils.visualization import plot_tsne
from norm_distribution import draw
# support for mini-batched forward
# only support one layer for one ctr_ntype
class MAGNN_nc_mb_layer(nn.Module):
    def __init__(self,
                 num_metapaths,
                 num_edge_type,
                 etypes_list,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 attn_drop=0.5):
        super(MAGNN_nc_mb_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # etype-specific parameters
        r_vec = None
        if rnn_type == 'TransE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim)))
        elif rnn_type == 'TransE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim)))
        elif rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim // 2, 2)))
        elif rnn_type == 'RotatE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim // 2, 2)))
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)

        # ctr_ntype-specific layers
        self.ctr_ntype_layer = MAGNN_ctr_ntype_specific(num_metapaths,
                                                        etypes_list,
                                                        in_dim,
                                                        num_heads,
                                                        attn_vec_dim,
                                                        rnn_type,
                                                        r_vec,
                                                        attn_drop,
                                                        use_minibatch=True)

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        self.fc = nn.Linear(in_dim * num_heads, out_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, inputs):
        # ctr_ntype-specific layers
        h, attn = self.ctr_ntype_layer(inputs)

        h_fc = self.fc(h)
        return h_fc, h, attn


class MAGNN_nc_mb_AC(nn.Module):
    def __init__(self,
                 num_metapaths,
                 num_edge_type,
                 etypes_list,
                 in_dims,
                 emb_dim,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 dropout_rate=0.5,
                 cuda=False,
                 feat_opt=None):
        super(MAGNN_nc_mb_AC, self).__init__()
        self.emb_dim = emb_dim
        self.feat_opt = feat_opt
        self.hidden_dim = hidden_dim #64
        self.in_dims = in_dims

        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(m, hidden_dim, bias=True) for m in in_dims])

        MD = nn.ModuleDict
        self.encoder1, self.encoder2 = MD({}), MD({})
        for i in range(len(in_dims)):
            if in_dims[i] <= hidden_dim:
                self.encoder1[str(i)] = nn.Linear(in_dims[i], hidden_dim, bias=True)
                #nn.init.xavier_normal_(self.encoder1[str(i)].weight, gain=1.414)
            else:
                self.encoder2[str(i)] = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=True),
                                                       nn.Linear(in_dims[i] - hidden_dim, hidden_dim,
                                                                 bias=True)])
                # nn.init.xavier_normal_(self.encoder2[str(i)][0].weight, gain=1.414)
                # nn.init.xavier_normal_(self.encoder2[str(i)][1].weight, gain=1.414)

        # !contrast loss
        self.contrast_loss = Contrast(hidden_dim, in_dims)

        # attribute completion layer
        self.hgnn_ac = HGNN_AC(in_dim=emb_dim, hidden_dim=attn_vec_dim, dropout=dropout_rate,
                               activation=F.elu, num_heads=num_heads, cuda=cuda)

        # feature dropout after attribute completion
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # MAGNN_nc_mb layers
        self.layer1 = MAGNN_nc_mb_layer(num_metapaths,
                                        num_edge_type,
                                        etypes_list,
                                        hidden_dim,
                                        out_dim,
                                        num_heads,
                                        attn_vec_dim,
                                        rnn_type,
                                        attn_drop=dropout_rate)


    def forward(self, inputs1, inputs2, epoch, flag):
        adj, feat_list, emb, mask_list, feat_keep_idx, feat_drop_idx, node_type_src = inputs1
        g_list, type_mask, edge_metapath_indices_list, target_idx_list = inputs2

        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=adj.device)

        for i in range(len(self.in_dims)):
            node_indices = np.where(type_mask == i)[0]
            if self.in_dims[i] <= self.hidden_dim:
                temp = F.elu(self.feat_drop(self.encoder1[str(i)](feat_list[i])))
                #temp_set.append(temp)
            else:
                # feats_std = []
                # for j in range(feat_list[i].shape[-1]):
                #     a = feat_list[i][:, j].softmax(dim=-1).log()
                #     b = torch.FloatTensor([1/torch.mean(feat_list[i][:, j]) for u in range(feat_list[i].shape[0])]).softmax(dim=-1).to(feat_list[i].device)
                #     feats_std.append(F.kl_div(a, b))

                feats_std = torch.std(feat_list[i].softmax(dim=0), dim=0)
                feats_std_topk, topk_indices = torch.topk(feats_std, self.hidden_dim, largest=True, sorted=False)
                indices_del = [i for i in torch.arange(feat_list[i].shape[1]).to(feat_list[i].device) if i not in topk_indices]
                feats_del = feat_list[i][:, indices_del]
                # draw(feats_del)
                # draw(feat_list[i][:, topk_indices])
                temp = self.encoder2[str(i)][0](feat_list[i][:, topk_indices]) + F.elu(self.encoder2[str(i)][1](feats_del))
                #temp_set.append(temp)
            transformed_features[node_indices] = temp
        # #draw(transformed_features)
        # for i, fc in enumerate(self.fc_list):
        #     node_indices = np.where(type_mask == i)[0]
        #     transformed_features[node_indices] = fc(feat_list[i])

        #contrast loss computation
        contrast_loss = self.contrast_loss(transformed_features, type_mask)
        feat_src = transformed_features
        # attribute completion
        feature_src_re = self.hgnn_ac(adj[mask_list[node_type_src]][:, mask_list[node_type_src]][:, feat_keep_idx],
                                      emb[mask_list[node_type_src]], emb[mask_list[node_type_src]][feat_keep_idx],
                                      feat_src[mask_list[node_type_src]][feat_keep_idx])
        loss_ac = F.mse_loss(feat_src[mask_list[node_type_src]][feat_drop_idx], feature_src_re[feat_drop_idx, :])

        for i, opt in enumerate(self.feat_opt):
            if opt == 1:
                feat_ac = self.hgnn_ac(adj[mask_list[i]][:, mask_list[node_type_src]],
                                       emb[mask_list[i]], emb[mask_list[node_type_src]],
                                       feat_src[mask_list[node_type_src]])
                transformed_features[mask_list[i]] = feat_ac

        transformed_features = self.feat_drop(transformed_features)
        # if flag == 'test' and epoch == 15:
        #     plot_tsne(transformed_features, type_mask, self.in_dims, epoch, flag)
        # hidden layers
        logits, h, attn = self.layer1((g_list, transformed_features, type_mask, edge_metapath_indices_list, target_idx_list))

        return logits, h, attn, loss_ac, contrast_loss

class Contrast(nn.Module):
    def __init__(self, hidden_dim, in_dims, tau=0.8, lam=0.5):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.in_dims = in_dims
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        eps = 1e-8
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)


        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)

        # z1_norm = z1_norm / torch.max(z1_norm, eps * torch.ones_like(z1_norm))
        # z2_norm = z2_norm / torch.max(z2_norm, eps * torch.ones_like(z2_norm))
        # sim_matrix = torch.mm(z1_norm, z2_norm.transpose(0, 1))
        return sim_matrix

    def forward(self, features, type_mask):
        contrast_loss = torch.zeros([]).to(features.device)
        #features = self.proj(features)
        for i in range(len(self.in_dims)):
            node_indices = np.where(type_mask == i)[0]
            positive = features[node_indices]
            negative = torch.cat([features[np.where(type_mask == t)[0]] for t in range(len(self.in_dims)) if t != i])
            positive_sim = self.sim(positive, positive)
            negative_sim = self.sim(positive, negative)
            log_positive = -torch.log(torch.sum(positive_sim))
            log_negative = -torch.log(torch.sum(negative_sim))
            contrast_loss = contrast_loss + log_positive - log_negative
        return 0.002*contrast_loss