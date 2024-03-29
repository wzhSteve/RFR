import time
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
from utils.pytorchtools import EarlyStopping
from utils.data import load_ACM_data
from utils.tools import index_generator, evaluate_results_nc, parse_minibatch, parse_mask
from model.MAGNN_nc_mb import MAGNN_nc_mb_AC

ap = argparse.ArgumentParser(description='MAGNN-AC testing for the ACM dataset')
ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
ap.add_argument('--batch-size', type=int, default=32, help='Batch size. Default is 8.')
ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')
ap.add_argument('--save-postfix', default='ACM', help='Postfix for the saved model and result. Default is DBLP.')
ap.add_argument('--feats-opt', type=str, default='011', help='010 means 1 type nodes use our processed feature.')
ap.add_argument('--feats-drop-rate', type=float, default=0.7, help='The ratio of attributes to be dropped.')
ap.add_argument('--loss-lambda', type=float, default=0.2, help='Coefficient lambda to balance loss.')
ap.add_argument('--cuda', default=True, help='Using GPU or not.')
args = ap.parse_args()
print(args)

hidden_dim = args.hidden_dim
num_heads = args.num_heads
attn_vec_dim = args.attn_vec_dim
rnn_type = args.rnn_type
num_epochs = args.epoch
patience = args.patience
batch_size = args.batch_size
neighbor_samples = args.samples
repeat = args.repeat
save_postfix = args.save_postfix
feats_opt = args.feats_opt
feats_drop_rate = args.feats_drop_rate
loss_lambda = args.loss_lambda
is_cuda = args.cuda

# random seed
seed = 123
random.seed()
np.random.seed()
# torch.manual_seed(seed)
# if is_cuda:
#     print('Using CUDA')
#     torch.cuda.manual_seed(seed)

feats_opt = list(feats_opt)
feats_opt = list(map(int, feats_opt))
print('feats_opt: {}'.format(feats_opt))

# 0 for papers, 1 for authors, 2 for subjects   0-PA 1-AP 2-PS 3-SP
etypes_list = [[0, 1], [2, 3]]
num_metapaths = 2
num_edge_type = 4
src_node_type = 0

# Params
out_dim = 3
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001
device = torch.device('cuda:0' if is_cuda else 'cpu')

# load data
adjlists, edge_metapath_indices_list, features_list, emb, adjM, type_mask, labels, train_val_test_idx = load_ACM_data()
features_list = [torch.FloatTensor(features).to(device) for features in features_list]
in_dims = [features.shape[1] for features in features_list]
emb_dim = emb.shape[1]
emb = torch.FloatTensor(emb).to(device)
adjM = torch.FloatTensor(adjM).to(device)
labels = torch.LongTensor(labels).to(device)
train_idx = train_val_test_idx['train_idx']
train_idx = np.sort(train_idx)
val_idx = train_val_test_idx['val_idx']
val_idx = np.sort(val_idx)
test_idx = train_val_test_idx['test_idx']
test_idx = np.sort(test_idx)

# train_len = train_idx.shape[0]
# val_len = val_idx.shape[0]
# test_len = test_idx.shape[0]
#
# type0 = 1436
# type1 = 3997
# labels_index = [i for i in range(labels.shape[0])]
# train_index_0 = random.sample(labels_index[:type0], 143)
# train_index_1 = random.sample(labels_index[type0:type1], 255)
# train_index_2 = random.sample(labels_index[type1:], 2)
# train_idx = train_index_0 + train_index_1 + train_index_2
# train_idx = np.sort(np.array(train_idx))
# val_index_0 = random.sample(labels_index[:type0], 143)
# val_index_1 = random.sample(labels_index[type0:type1], 255)
# val_index_2 = random.sample(labels_index[type1:], 2)
# val_idx = val_index_0 + val_index_1 + val_index_2
# val_idx = np.sort(np.array(val_idx))
# test_index_0 = random.sample(labels_index[:type0], 1150)
# test_index_1 = random.sample(labels_index[type0:type1], 2051)
# test_index_2 = random.sample(labels_index[type1:], 18)
# test_idx = test_index_0 + test_index_1 + test_index_2
# test_idx = np.sort(np.array(test_idx))
#
# labels[[i for i in range(type0)]] = 0
# labels[[ i for i in range(type0, type1)]] = 1
# labels[[ i for i in range(type1, len(labels))]] = 2


print('data load finish')
cnt_0 = 0
cnt_1 = 0
cnt_2 = 0
for i in train_idx:
    if labels[i] == 0:
        cnt_0 = cnt_0 + 1
    elif labels[i] == 1:
        cnt_1 = cnt_1 + 1
    elif labels[i] == 2:
        cnt_2 = cnt_2 + 1
print('train_data type 0: {:.2f}%, type 1: {:.2f}%, type 2: {:.2f}%  node_num:type 0: {}, type 1: {}, type 2: {}'.format(100*cnt_0/(cnt_0 + cnt_1 + cnt_2), 100*cnt_1/(cnt_0 + cnt_1 + cnt_2), 100*cnt_2/(cnt_0 + cnt_1 + cnt_2), cnt_0, cnt_1, cnt_2))

cnt_0 = 0
cnt_1 = 0
cnt_2 = 0
for i in val_idx:
    if labels[i] == 0:
        cnt_0 = cnt_0 + 1
    elif labels[i] == 1:
        cnt_1 = cnt_1 + 1
    elif labels[i] == 2:
        cnt_2 = cnt_2 + 1
print('valid_data type 0: {:.2f}%, type 1: {:.2f}%, type 2: {:.2f}%  node_num:type 0: {}, type 1: {}, type 2: {}'.format(100*cnt_0/(cnt_0 + cnt_1 + cnt_2), 100*cnt_1/(cnt_0 + cnt_1 + cnt_2), 100*cnt_2/(cnt_0 + cnt_1 + cnt_2), cnt_0, cnt_1, cnt_2))
cnt_0 = 0
cnt_1 = 0
cnt_2 = 0
for i in test_idx:
    if labels[i] == 0:
        cnt_0 = cnt_0 + 1
    elif labels[i] == 1:
        cnt_1 = cnt_1 + 1
    elif labels[i] == 2:
        cnt_2 = cnt_2 + 1
print('test_data type 0: {:.2f}%, type 1: {:.2f}%, type 2: {:.2f}%  node_num:type 0: {}, type 1: {}, type 2: {}'.format(100*cnt_0/(cnt_0 + cnt_1 + cnt_2), 100*cnt_1/(cnt_0 + cnt_1 + cnt_2), 100*cnt_2/(cnt_0 + cnt_1 + cnt_2), cnt_0, cnt_1, cnt_2))


#7组 macro和micro
svm_macro_avg = np.zeros((7, ), dtype=np.float)
svm_micro_avg = np.zeros((7, ), dtype=np.float)
nmi_avg = 0
ari_avg = 0
print('start train with repeat = {}\n'.format(repeat))
for cur_repeat in range(repeat):
    print('cur_repeat = {}   ==============================================================='.format(cur_repeat))
    net = MAGNN_nc_mb_AC(num_metapaths, num_edge_type, etypes_list, in_dims, emb_dim, hidden_dim, out_dim,
                         num_heads, attn_vec_dim, rnn_type, dropout_rate, is_cuda, feats_opt)

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    print('model init finish\n')

    # training loop
    print('training...')
    net.train()
    early_stopping = EarlyStopping(patience=patience, verbose=True,
                                   save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
    train_idx_generator = index_generator(batch_size=batch_size, indices=train_idx)
    val_idx_generator = index_generator(batch_size=batch_size, indices=val_idx, shuffle=False)
    for epoch in range(num_epochs):
        # training
        t = time.time()
        net.train()
        train_loss_avg = 0
        for iteration in range(train_idx_generator.num_iterations()): #16
            # forward
            train_idx_batch = train_idx_generator.next()
            train_idx_batch.sort()

            train_g_list, train_indices_list, train_idx_batch_mapped_list = parse_minibatch(
                adjlists, edge_metapath_indices_list, train_idx_batch, device, neighbor_samples)
            mask_list, feat_keep_idx, feat_drop_idx = parse_mask(
                indices_list=train_indices_list, type_mask=type_mask, num_classes=out_dim,
                src_type=src_node_type, rate=feats_drop_rate, device=device)

            logits, _, _, loss_ac, contrast_loss = net(
                (adjM, features_list, emb, mask_list, feat_keep_idx, feat_drop_idx, src_node_type),
                (train_g_list, type_mask, train_indices_list, train_idx_batch_mapped_list), iteration, 'train')
            logp = F.log_softmax(logits, 1)
            loss_classification = F.nll_loss(logp, labels[train_idx_batch])

            train_loss = loss_classification + loss_lambda * loss_ac + contrast_loss
            train_loss_avg += loss_classification.item()
            # auto grad
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        train_loss_avg /= train_idx_generator.num_iterations()
        train_time = time.time() - t

        # validation
        t = time.time()
        net.eval()
        val_logp = []
        val_loss_avg = 0
        with torch.no_grad():
            for iteration in range(val_idx_generator.num_iterations()):
                # forward
                val_idx_batch = val_idx_generator.next()
                val_g_list, val_indices_list, val_idx_batch_mapped_list = parse_minibatch(
                    adjlists, edge_metapath_indices_list, val_idx_batch, device, neighbor_samples)
                mask_list, feat_keep_idx, feat_drop_idx = parse_mask(
                    indices_list=val_indices_list, type_mask=type_mask, num_classes=out_dim,
                    src_type=src_node_type, rate=feats_drop_rate, device=device)

                logits, _, _, loss_ac, contrast_loss = net(
                    (adjM, features_list, emb, mask_list, feat_keep_idx, feat_drop_idx, src_node_type),
                    (val_g_list, type_mask, val_indices_list, val_idx_batch_mapped_list), iteration, 'vali')
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp, labels[val_idx_batch])
                val_loss_avg += val_loss.item()+ contrast_loss
            val_loss_avg /= val_idx_generator.num_iterations()

        val_time = time.time() - t
        print(
            'Epoch {:05d} | Train_Loss {:.4f} | Train_Time(s) {:.4f} | Val_Loss {:.4f} | Val_Time(s) {:.4f}'.format(
                epoch, train_loss_avg, train_time, val_loss_avg, val_time))

        # early stopping
        early_stopping(val_loss_avg, net)
        if early_stopping.early_stop:
            print('Early stopping!')
            break

    # testing with evaluate_results_nc
    print('\ntesting...')
    test_idx_generator = index_generator(batch_size=batch_size, indices=test_idx, shuffle=False)
    net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
    net.eval()
    test_embeddings = []
    with torch.no_grad():
        for iteration in range(test_idx_generator.num_iterations()): #16
            # forward
            test_idx_batch = test_idx_generator.next()
            test_g_list, test_indices_list, test_idx_batch_mapped_list = parse_minibatch(
                adjlists, edge_metapath_indices_list, test_idx_batch, device, neighbor_samples)
            mask_list, feat_keep_idx, feat_drop_idx = parse_mask(
                indices_list=test_indices_list, type_mask=type_mask, num_classes=out_dim,
                src_type=src_node_type, rate=feats_drop_rate, device=device)
            _, embeddings, _, _, _ = net(
                (adjM, features_list, emb, mask_list, feat_keep_idx, feat_drop_idx, src_node_type),
                (test_g_list, type_mask, test_indices_list, test_idx_batch_mapped_list), iteration, 'test')
            test_embeddings.append(embeddings)

        test_embeddings = torch.cat(test_embeddings, 0)
        embeddings = test_embeddings.detach().cpu().numpy()
        svm_macro, svm_micro, nmi, ari = evaluate_results_nc(embeddings, labels[test_idx].cpu().numpy(), out_dim)
        svm_macro_avg = svm_macro_avg + svm_macro
        svm_micro_avg = svm_micro_avg + svm_micro
        nmi_avg += nmi
        ari_avg += ari

svm_macro_avg = svm_macro_avg / repeat
svm_micro_avg = svm_micro_avg / repeat
nmi_avg /= repeat
ari_avg /= repeat
print('---\nThe average of {} results:'.format(repeat))
print('Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]))
print('Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]))
print('NMI: {:.6f}'.format(nmi_avg))
print('ARI: {:.6f}'.format(ari_avg))
print('all finished')

with open('log-ACM-statistics.txt', 'a+') as f:
    f.writelines('\n' + 'Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]) + '\n' +
                 'Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]) + '\n')
