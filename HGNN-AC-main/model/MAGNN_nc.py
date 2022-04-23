import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.base_MAGNN import MAGNN_ctr_ntype_specific
from model.HGNN_AC import HGNN_AC

fc_switch = False


# multi-layer support
class MAGNN_nc_layer(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 attn_drop=0.5):
        super(MAGNN_nc_layer, self).__init__()
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
        self.ctr_ntype_layers = nn.ModuleList()
        for i in range(len(num_metapaths_list)):
            self.ctr_ntype_layers.append(MAGNN_ctr_ntype_specific(num_metapaths_list[i],
                                                                  etypes_lists[i],
                                                                  in_dim,
                                                                  num_heads,
                                                                  attn_vec_dim,
                                                                  rnn_type,
                                                                  r_vec,
                                                                  attn_drop,
                                                                  use_minibatch=False))

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        if fc_switch:
            self.fc1 = nn.Linear(in_dim, out_dim, bias=False)
            self.fc2 = nn.Linear(in_dim * num_heads, out_dim, bias=True)
            nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
            nn.init.xavier_normal_(self.fc2.weight, gain=1.414)
        else:
            self.fc = nn.Linear(in_dim * num_heads, out_dim, bias=True)
            nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features, type_mask, edge_metapath_indices_lists = inputs

        # ctr_ntype-specific layers
        h = torch.zeros(type_mask.shape[0], self.in_dim * self.num_heads, device=features.device)
        for i, (g_list, edge_metapath_indices_list, ctr_ntype_layer) in enumerate(zip(g_lists, edge_metapath_indices_lists, self.ctr_ntype_layers)):
            h[np.where(type_mask == i)[0]], beta = ctr_ntype_layer((g_list, features, type_mask, edge_metapath_indices_list))

        if fc_switch:
            h_fc = self.fc1(features) + self.fc2(h)
        else:
            h_fc = self.fc(h)
        return h_fc, h, beta


class MAGNN_nc_AC(nn.Module):
    def __init__(self,
                 num_layers,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
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
        super(MAGNN_nc_AC, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.feat_opt = feat_opt
        self.in_dims = in_dims

        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(m, hidden_dim, bias=True) for m in in_dims])

        MD = nn.ModuleDict
        self.encoder1, self.encoder2 = MD({}), MD({})
        for i in range(len(in_dims)):
            if in_dims[i] <= hidden_dim:
                self.encoder1[str(i)] = nn.Linear(in_dims[i], hidden_dim, bias=True)
            else:
                self.encoder2[str(i)] = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=True),
                                                       nn.Linear(in_dims[i] - hidden_dim, hidden_dim,
                                                                 bias=True)])
        # !contrast loss
        self.contrast_loss = Contrast(hidden_dim, in_dims)

        self.hgnn_ac = HGNN_AC(in_dim=emb_dim, hidden_dim=attn_vec_dim, dropout=dropout_rate,
                               activation=F.elu, num_heads=num_heads, cuda=cuda)

        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # feature dropout
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x

        # MAGNN_nc layers
        self.layers = nn.ModuleList()
        # hidden layers
        for l in range(num_layers - 1):
            self.layers.append(MAGNN_nc_layer(num_metapaths_list, num_edge_type, etypes_lists, hidden_dim, hidden_dim,
                                              num_heads, attn_vec_dim, rnn_type, attn_drop=dropout_rate))
        # output projection layer
        self.layers.append(MAGNN_nc_layer(num_metapaths_list, num_edge_type, etypes_lists, hidden_dim, out_dim,
                                          num_heads, attn_vec_dim, rnn_type, attn_drop=dropout_rate))

    def forward(self, inputs_1, inputs_2, target_node_indices, epoch, flag):
        adj, feat_list, emb, mask_list, feat_keep_idx, feat_drop_idx, node_type_src = inputs_1
        g_lists, type_mask, edge_metapath_indices_lists = inputs_2

        # ntype-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=feat_list[0].device)

        for i in range(len(self.in_dims)):
            node_indices = np.where(type_mask == i)[0]
            if self.in_dims[i] <= self.hidden_dim:
                temp = F.elu(self.feat_drop(self.encoder1[str(i)](feat_list[i])))
            else:
                feats_std = torch.std(feat_list[i].softmax(dim=0), dim=0)
                feats_std_topk, topk_indices = torch.topk(feats_std, self.hidden_dim, largest=True, sorted=False)
                indices_del = [i for i in torch.arange(feat_list[i].shape[1]).to(feat_list[i].device) if i not in topk_indices]
                feats_del = feat_list[i][:, indices_del]
                temp = self.encoder2[str(i)][0](feat_list[i][:, topk_indices]) + F.elu(self.encoder2[str(i)][1](feats_del))
            transformed_features[node_indices] = temp

        # for i, fc in enumerate(self.fc_list):
        #     node_indices = np.where(type_mask == i)[0]
        #     transformed_features[node_indices] = fc(feat_list[i])

        # contrast_loss = self.contrast_loss(transformed_features, type_mask)

        feat_src = transformed_features
        # attribute completion
        feat_src_re = self.hgnn_ac(adj[mask_list[node_type_src]][:, mask_list[node_type_src]][:, feat_keep_idx],
                                   emb[mask_list[node_type_src]], emb[mask_list[node_type_src]][feat_keep_idx],
                                   feat_src[mask_list[node_type_src]][feat_keep_idx])
        loss_ac = F.mse_loss(feat_src[mask_list[node_type_src]][feat_drop_idx], feat_src_re[feat_drop_idx, :])

        for i, opt in enumerate(self.feat_opt):
            if opt == 1:
                feat_ac = self.hgnn_ac(adj[mask_list[i]][:, mask_list[node_type_src]],
                                       emb[mask_list[i]], emb[mask_list[node_type_src]],
                                       feat_src[mask_list[node_type_src]])
                transformed_features[mask_list[i]] = feat_ac
        h = self.feat_drop(transformed_features)

        # hidden layers
        for l in range(self.num_layers - 1):
            h, _, _ = self.layers[l]((g_lists, h, type_mask, edge_metapath_indices_lists))
            h = F.elu(h)
        # output projection layer
        logits, h, beta = self.layers[-1]((g_lists, h, type_mask, edge_metapath_indices_lists))

        # return only the target nodes' logits and embeddings
        return logits[target_node_indices], h[target_node_indices], beta, loss_ac, 0#contrast_loss

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
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, features, type_mask):
        contrast_loss = torch.zeros([]).to(features.device)
        #features = self.proj(features)
        for i in range(len(self.in_dims)):
            node_indices = np.where(type_mask == i)[0]
            positive = features[node_indices]
            negative = torch.cat([features[np.where(type_mask == t)[0]] for t in range(len(self.in_dims)) if t != i])
            # positive = self.proj(positive)
            # negative = self.proj(negative)
            positive_sim = self.sim(positive, positive)
            negative_sim = self.sim(positive, negative)
            log_positive = -torch.log(torch.sum(positive_sim))
            log_negative = -torch.log(torch.sum(negative_sim))
            contrast_loss = contrast_loss + log_positive - log_negative
        return 0.02 * contrast_loss