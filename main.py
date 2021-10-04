import os
import time
from typing import List

import torch
import argparse

from model import TAT4Rec
from tqdm import tqdm
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
# training hyperparameters
parser.add_argument('--dataset', default='Steam')
parser.add_argument('--train_dir', default='123234')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--dropout_rate_encoder',default=0.2,type=float)
parser.add_argument('--dropout_rate_decoder',default=0.4,type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--if_use_lr_scheduler',default=True)
parser.add_argument('--inference_only', default=False, type=str2bool) # 暂时无用
parser.add_argument('--state_dict_path', default=None, type=str) # 暂时无用

# model hyperparameters
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=100, type=int)

parser.add_argument('--hidden_units_item',default=100)

parser.add_argument('--lag_time_bins',default=1024)
parser.add_argument('--lagtime_window_size',default=20)
parser.add_argument('--encoder_blocks_num',default=2)
parser.add_argument('--encoder_heads_num',default=2)
parser.add_argument('--decoder_blocks_num',default=2)
parser.add_argument('--decoder_heads_num',default=1)
parser.add_argument('--time_lag_first',default=True)
parser.add_argument('--if_point_wise_feedforward',default=False)
parser.add_argument('--if_T_fixup',default=True)
parser.add_argument('--if_gelu',default=False)
parser.add_argument('--scale_coefficient',default=8)
parser.add_argument('--if_shift_time',default=False)


args = parser.parse_args()

if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

dataset = data_partition(args.dataset,args.maxlen)
[user_train, user_valid, user_test, usernum, itemnum,user_latest_time,user_first_time] = dataset # 注意此处均为字典
num_batch = len(user_train) // args.batch_size

cc = 0.0
time_avg_gap = 0
max_time_lag = 0
time_per_interaction = 0

max_u = 0
for u in user_train: #统计数据集相关特征
    cc += len(user_train[u])
    #print('time_gap:',user_train[u][-1][1] - user_train[u][0][1])
    time_avg_gap += user_latest_time[u] - user_train[u][0][1]
    time_per_interaction += (user_latest_time[u] - user_train[u][0][1])/len(user_train[u])
    if user_latest_time[u] - user_train[u][0][1] > max_time_lag:
        max_time_lag = max(user_latest_time[u] - user_train[u][0][1],max_time_lag)
        max_u = u
print('average sequence length: %.2f' % (cc / len(user_train)))
print('averge user timestamp gap: %.2f' % (time_avg_gap/len(user_train))) # 86400
print('max time lag: %.2f with max_u:%d' % (max_time_lag,max_u))
print('time_per_interaction:',time_per_interaction/len(user_train)) # 86400
print('usernum:',usernum,'itemnum:',itemnum)
#print('len train:',len(user_train))
if args.dataset == 'Userbehavior':
    max_time_lag = 669973
max_time_lag = max_time_lag/args.scale_coefficient
print('max time lag:',max_time_lag)

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
if __name__ == '__main__':

    sampler = WarpSampler(user_train, usernum, itemnum, user_latest_time = user_latest_time,
                          user_first_time=user_first_time,batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3,
                          time_lag_first=args.time_lag_first,if_shift_time=args.if_shift_time)
    model = TAT4Rec(usernum, itemnum, args).to(args.device)
    #print(model)
    if args.if_T_fixup:
        parameters_init_T_fixup(model,args.encoder_blocks_num,args.decoder_blocks_num,args.hidden_units)

    #model.eval()

    #print('Evaluating', end='')
    #t_start_test = time.time()
    #t_test = evaluate(model, dataset, args, max_time_lag, args.time_lag_first)
    #t_end_test = time.time()
    #print('total_test_time', t_end_test - t_start_test)



    model.train()  # enable model training
    epoch_start_idx = 1
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    #scheduler = torch.optim.lr_scheduler.StepLR(adam_optimizer,gamma=0.2,step_size=100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam_optimizer,T_max=args.num_epochs)

    T = 0.0
    t0 = time.time()
    best_average_score = 0.0
    min_loss = 100


    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        t_start = time.time() # 统计epoch开始时间

        if args.inference_only: break  # just to decrease identition
        average_loss = 0.0
        for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, seq_ts, seq_lag_from_now, pos, neg, x_mask = sampler.next_batch()
            #print('u:',u)
            u, seq, seq_ts,seq_lag_from_now,pos, neg,x_mask = np.array(u), np.array(seq), np.array(seq_ts),np.array(seq_lag_from_now),np.array(pos), np.array(neg),np.array(x_mask)
            pos_logits, neg_logits = model(u, seq,seq_ts,seq_lag_from_now, pos, neg,x_mask,max_time_lag)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                   device=args.device)
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)

            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            if args.if_use_lr_scheduler:
                scheduler.step()
            average_loss += loss.item()
            #print("loss in epoch {} iteration {}: {}".format(epoch, step,
            #                                                 loss.item()))  # expected 0.4~0.6 after init few epochs
        curent_loss = average_loss/num_batch
        t_end = time.time()
        print('time_per_epoch(training):',t_end-t_start)

        print('Epoch:%d average loss:%.5f'%(epoch,curent_loss))

        if epoch % 1 == 0:#(epoch<200 and epoch % 20 == 0) or (epoch>=200 and curent_loss<min_loss):
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args, max_time_lag, args.time_lag_first)
            t_valid = evaluate_valid(model, dataset, args, max_time_lag, args.time_lag_first)
            print('epoch:%d, time: %f(s), valid (NDCG@5: %.4f, HR@5: %.4f), test (NDCG@5: %.4f, HR@5: %.4f)'
                  % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            average_score = (t_test[0] + t_test[1] + t_test[2] + t_test[3] + t_test[4]) / 5

            if best_average_score < average_score: # 保存最佳模型
                folder = args.dataset + '_' + args.train_dir
                fname = 'TAT4SRec_best.pth'
                torch.save(model.state_dict(),os.path.join(folder, fname))




            best_average_score = max(best_average_score, average_score)
            # f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.write(
                "epoch:%d valid (NDCG@5:%.6f HR@5:%.6f NDCG@10:%.6f HR@10:%.6f MRR:%.6f),test (NDCG@5:%.6f HR@5:%.6f NDCG@10:%.6f HR@10:%.6f MRR:%.6f as:%.5f loss:%.5f bas:%.5f)\n"
                % (
                    epoch, t_valid[0], t_valid[1], t_valid[2], t_valid[3], t_valid[4], t_test[0], t_test[1], t_test[2],
                    t_test[3],
                    t_test[4], average_score, average_loss / num_batch, best_average_score))
            f.flush()
            t0 = time.time()
            model.train()

        min_loss = min(min_loss,curent_loss)
        #if epoch == args.num_epochs:
        ##    folder = args.dataset + '_' + args.train_dir
        #    fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
        #    fname = fname.format(args.num_epochs, args.lr,  args.hidden_units,
        #                         args.maxlen)
        #    torch.save(model.state_dict(), os.path.join(folder, fname))


    f.close()
    sampler.close()
    print("Done")
