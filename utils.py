import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import re


# train/val/test data generation
def data_partition(fname,max_len=0):
    #max_len = 0
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    user_latest_time = {}
    user_first_time = {}

    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u,timestamp, i = line.rstrip().split(' ') # 读取用户编号和商品编号
        u = int(u)
        i = int(i)
        timestamp = int(timestamp)

        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append([i,timestamp]) # User:{user:(i,timestamp)}

    for user in User:
        nfeedback = len(User[user]) #获得每个用户总的购买次数
        if nfeedback < 3: # 长度小于3的，直接用作训练
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
            user_latest_time[user] = User[user][-1][1]
            user_first_time[user] = User[user][0][1]

        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2]) # 倒数第二个作为valid
            user_test[user] = []
            user_test[user].append(User[user][-1]) # 倒数第一个作为test
            user_latest_time[user] = User[user][-1][1]
            if len(User[user]) >= max_len:
                user_first_time[user] = User[user][-max_len][1]
            else:
                user_first_time[user] = User[user][0][1]
    return [user_train, user_valid, user_test, usernum, itemnum,user_latest_time,user_first_time]

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function_shift_timestamp(user_train, usernum, itemnum,user_latest_time, user_first_time,batch_size, maxlen, result_queue, SEED,time_lag_first=False):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        x_mask = np.zeros([maxlen], dtype=np.int32)  # 用来mask掉padding
        seq = np.zeros([maxlen], dtype=np.int32)
        seq_t = np.zeros([maxlen], dtype=np.int32)  # 新加入用以保存timestamp
        seq_lag_time_from_now = np.zeros([maxlen], dtype=np.int32)  # 新加入用以保存距离当前的时间
        seq_lag_time_from_first = np.zeros([maxlen], dtype=np.int32)  # 新加入用以保存距离第一次访问的时间

        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        nxt = user_train[user][-1][0]  # 最后一个购买的商品
        current_time = user_train[user][-1][1] # 当前购买时间
        idx = maxlen - 1

        ts = set([x[0] for x in user_train[user]])  # 该用户购买的所有商品编号

        for i in reversed(user_train[user][:-1]):  # i从倒数第二个商品编号开始
            # SSE for user side (2 lines)
            seq[idx] = i[0]
            #seq_t[idx] = i[1]
            seq_t[idx] = current_time
            seq_lag_time_from_now[idx] = user_latest_time[user] - current_time
            seq_lag_time_from_first[idx] = current_time - user_first_time[user]
            current_time = i[1]

            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)  # 在不是当前用户购买的商品中随机选取一个商品
            nxt = i[0]
            idx -= 1
            if idx == -1: break

        if len(user_train[user][:-1])<maxlen: # 修改lag from now的未填充部分
            for i in range(maxlen-len(user_train[user][:-1])):
                seq_lag_time_from_now[i] = seq_lag_time_from_first[maxlen-len(user_train[user][:-1])]

        x_mask[np.where(seq == 0)] = 1

        if time_lag_first:
            return (user, seq, seq_t, seq_lag_time_from_first, pos, neg, x_mask)
        else:
            return (user, seq, seq_t, seq_lag_time_from_now, pos, neg, x_mask)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())  # sample() == (user,seq,pos,neg)
        result_queue.put(zip(*one_batch))  # zip == (users,seqs,seq_ts,user_lag_time_form_nows,poses,negs)



def sample_function(user_train, usernum, itemnum,user_latest_time, user_first_time,batch_size, maxlen, result_queue, SEED,time_lag_first=False):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        x_mask = np.zeros([maxlen],dtype=np.int32) # 用来mask掉padding
        seq = np.zeros([maxlen], dtype=np.int32)
        seq_t = np.zeros([maxlen], dtype=np.int32) # 新加入用以保存timestamp
        #seq_t = np.full([maxlen], fill_value= user_train[user][0][1], dtype=np.int32)  # 新加入用以保存timestamp
        seq_lag_time_from_now = np.zeros([maxlen], dtype=np.int32) # 新加入用以保存距离当前的时间
        seq_lag_time_from_first = np.zeros([maxlen], dtype=np.int32)  # 新加入用以保存距离第一次访问的时间

        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)

        nxt = user_train[user][-1][0] # 最后一个购买的商品
        idx = maxlen - 1

        ts = set([x[0] for x in user_train[user]]) # 该用户购买的所有商品编号



        for i in reversed(user_train[user][:-1]): # i从倒数第二个商品编号开始
            # SSE for user side (2 lines)
            seq[idx] = i[0]
            seq_t[idx] = i[1]
            seq_lag_time_from_now[idx] = user_latest_time[user] - i[1]
            seq_lag_time_from_first[idx] = i[1] - user_first_time[user]

            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts) # 在不是当前用户购买的商品中随机选取一个商品
            nxt = i[0]
            idx -= 1
            if idx == -1: break

        x_mask[np.where(seq==0)] = 1

        if time_lag_first:
            return (user, seq, seq_t, seq_lag_time_from_first, pos, neg, x_mask)
        else:
            return (user, seq, seq_t, seq_lag_time_from_now, pos, neg, x_mask)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample()) # sample() == (user,seq,pos,neg)
        result_queue.put(zip(*one_batch))# zip == (users,seqs,seq_ts,user_lag_time_form_nows,poses,negs)

class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, user_latest_time,user_first_time,batch_size=64, maxlen=10, n_workers=1,time_lag_first = False,if_shift_time=False):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            if if_shift_time:
                self.processors.append(
                    Process(target=sample_function_shift_timestamp, args=(User,
                                                          usernum,
                                                          itemnum,
                                                          user_latest_time,
                                                          user_first_time,
                                                          batch_size,
                                                          maxlen,
                                                          self.result_queue,
                                                          np.random.randint(2e9),
                                                          time_lag_first
                                                          )))

            else:
                self.processors.append(
                    Process(target=sample_function, args=(User,
                                                          usernum,
                                                          itemnum,
                                                          user_latest_time,
                                                          user_first_time,
                                                          batch_size,
                                                          maxlen,
                                                          self.result_queue,
                                                          np.random.randint(2e9),
                                                          time_lag_first
                                                          )))

            self.processors[-1].daemon = True
            self.processors[-1].start() # 启动新建进程

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def evaluate(model, dataset, args,max_time_lag,time_lag_first=False):
    [train, valid, test, usernum, itemnum,user_latest_time,user_first_time] = copy.deepcopy(dataset)

    NDCG_5 = 0.0
    HT_5 = 0.0
    NDCG_10 = 0.0
    HT_10 = 0.0
    MRR = 0.0

    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users: # 随机选取小于10000个用户数据

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        seq_lag_time_from_now = np.zeros([args.maxlen], dtype=np.int32)  # 新加入用以保存距离当前的时间
        seq_lag_time_from_first = np.zeros([args.maxlen], dtype=np.int32)  # 新加入用以保存距离第一次访问的时间

        idx = args.maxlen - 1
        seq[idx] = valid[u][0][0] # seq最后一个设为valid
        seq_lag_time_from_now[idx] = user_latest_time[u] - valid[u][0][1]
        seq_lag_time_from_first[idx] = valid[u][0][1] - user_first_time[u]

        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0] # seq剩下的为train
            seq_lag_time_from_now[idx] =  user_latest_time[u] - i[1]
            seq_lag_time_from_first[idx] =  i[1] - user_first_time[u]

            idx -= 1
            if idx == -1: break
        rated = set([x[0] for x in train[u]]) # 用户购买的所有商品
        rated.add(0)

        item_idx = [test[u][0][0]] # 用户实际购买的最后一个商品
        for _ in range(100): # 总共搜索101个结果
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t) # 随机选取100个不在用户购买范围里的商品

        if time_lag_first:
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq_lag_time_from_first], item_idx]],
                                         max_time_lag)
        else:
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq_lag_time_from_now], item_idx]],
                                         max_time_lag)

        predictions = predictions[0] #  DESC [101]
        #print('predictions:',predictions)
        rank = predictions.argsort().argsort()[0].item() # 看目标商品能排第几个

        valid_user += 1
        if rank < 5: # 新加入
            NDCG_5 += 1 / np.log2(rank + 2)
            HT_5 += 1
        MRR += 1/(rank+1)

        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG_5 / valid_user, HT_5 / valid_user, NDCG_10 / valid_user, HT_10 / valid_user, MRR / valid_user # 归一化


# evaluate on val set
def evaluate_valid(model, dataset, args,max_time_lag,time_lag_first=False):
    [train, valid, test, usernum, itemnum,user_latest_time,user_first_time] = copy.deepcopy(dataset)

    NDCG_5 = 0.0
    HT_5 = 0.0
    NDCG_10 = 0.0
    HT_10 = 0.0
    MRR = 0.0
    valid_user = 0.0

    if usernum>10000:  # 注意这里只选取10000个用户进行验证
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue


        seq = np.zeros([args.maxlen], dtype=np.int32)
        seq_lag_time_from_now = np.zeros([args.maxlen], dtype=np.int32)  # 新加入用以保存距离当前的时间
        seq_lag_time_from_first = np.zeros([args.maxlen], dtype=np.int32)  # 新加入用以保存距离第一次访问的时间

        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            seq_lag_time_from_now[idx] = user_latest_time[u] - i[1]
            seq_lag_time_from_first[idx] = i[1] - user_first_time[u]

            idx -= 1
            if idx == -1: break

        rated = set([x[0] for x in train[u]])
        rated.add(0)
        item_idx = [valid[u][0][0]] # 最后的目标预测商品为valid
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        if time_lag_first:
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [seq_lag_time_from_first], item_idx]],
                                         max_time_lag)
        else:
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq],[seq_lag_time_from_now], item_idx]],max_time_lag)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 5: # 新加入
            NDCG_5 += 1 / np.log2(rank + 2)
            HT_5 += 1
        MRR += 1/(rank+1)

        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG_5 / valid_user, HT_5 / valid_user, NDCG_10 / valid_user, HT_10 / valid_user, MRR/valid_user


def trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization (approximation)"
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)

def parameters_init_T_fixup(model,encoder_layer_num,decoder_layer_num,embedding_size):
    print('model:',model)
    for name, param in model.named_parameters():
        if re.match(r'.*bias$|.*bn\.weight$|.*norm.*\.weight', name):
            # print('can not init: ',name)
            #cannot_init.append(name)
            continue
        gain = 1
        if re.match(r'.*decoder.*', name):
            gain = (9 * decoder_layer_num) ** (-1. / 4.)
            #decoder.append(name)
            if re.match(f'.*in_proj_weight$', name):
                #in_proj_weight.append(name)
                gain *= (2 ** 0.5)
        elif re.match(r'.*encoder.*', name):
            #encoder.append(name)
            if re.match(f'.*in_proj_weight$', name):  # gain *= (2**0.5)
                #in_proj_weight.append(name)
                gain *= (2 ** 0.5)

        if re.match(r'^lag_from_now_emb|^pos_emb|^item_emb|^user_emb', name):
            #embedding.append(name)
            trunc_normal_(param.data,std=(4.5*(encoder_layer_num+decoder_layer_num))**(-1./4.)*embedding_size**(-0.5))
        else:
            torch.nn.init.xavier_normal_(param.data, gain=gain)