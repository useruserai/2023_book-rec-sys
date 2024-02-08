import pandas as pd
import numpy as np
import time
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
import tensorflow as tf
from torch.utils.data import DataLoader
import itertools
import os
import operator

from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares

from collections import defaultdict
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchtools.optim import Ranger
# from ranger import Ranger
from adabelief_pytorch import AdaBelief

import adabelief_pytorch
import tqdm
import random
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity as cs
import seaborn as sns

device = torch.device('cuda:0')
print(torch.version.cuda)

"""
    1. Data Preprocessing for DDPG algorithm
"""

df = pd.read_csv('C:/Users/Catholic/PycharmProjects/pythonProject/thebest52-master/data/df_merged.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
print(df)
print(time.time())

df_cat_title = df.drop(['user', 'rating'], axis=1)

df_cat_title.duplicated().value_counts()

df_cat_title.drop_duplicates(subset='item', inplace=True)

df_cat_title.set_index('item', inplace=True)

#
# """
#     2. indexing and embedding
# """
# # 고유한 유저, 아티스트를 찾아내는 코드

'''
df_origin = pd.read_csv('C:/Users/Catholic/PycharmProjects/pythonProject/thebest52-master/data/df_merged.csv')
#
user_unique = df_origin['user'].unique()
item_unique = df_origin['item'].unique()
num_user = df_origin['user'].nunique()
num_item = df_origin['item'].nunique()
print(user_unique, item_unique, num_user, num_item)
#
# # 유저, 아티스트 indexing 하는 코드 idx는 index의 약자입니다.
user_to_idx = {v:k for k,v in enumerate(user_unique)}
idx_to_user = {k:v for k,v in enumerate(user_unique)}
item_to_idx = {v:k for k,v in enumerate(item_unique)}
idx_to_item = {k:v for k,v in enumerate(item_unique)}
print(user_to_idx)
#
#
# # indexing을 통해 데이터 컬럼 내 값을 바꾸는 코드
# # user_to_idx.get을 통해 user_id 컬럼의 모든 값을 인덱싱한 Series를 구해 봅시다.
# # 혹시 정상적으로 인덱싱되지 않은 row가 있다면 인덱스가 NaN이 될 테니 dropna()로 제거합니다.
temp_user_data = df_origin['user'].map(user_to_idx.get).dropna()
if len(temp_user_data) == len(df_origin):   # 모든 row가 정상적으로 인덱싱되었다면
    print('userId column indexing OK!!')
    df_origin['user'] = temp_user_data   # data['userId']을 인덱싱된 Series로 교체해 줍니다.
else:
    print('userId column indexing Fail!!')
#
# # movie_to_idx을 통해 artist 컬럼도 동일한 방식으로 인덱싱해 줍니다.
temp_item_data = df_origin['item'].map(item_to_idx.get).dropna()
if len(temp_item_data) == len(df_origin):
    print('itemId column indexing OK!!')
    df_origin['item'] = temp_item_data
else:
    print('itemId column indexing Fail!!')
#
# #user x item
csr_data = csr_matrix((df_origin['rating'], (df_origin['user'], df_origin['item'])), shape= (num_user, num_item))
print("csr_data", type(csr_data), csr_data)
#
als_model = AlternatingLeastSquares(factors=100, regularization=0.01, use_gpu=False,
                                    iterations=15, dtype=np.float32, calculate_training_loss=True, num_threads=1)
#
# #item x user
csr_data_transpose = csr_data.T
als_model.fit(csr_data)
#
item_embeddings_dict = {idx_to_item[i]:tf.convert_to_tensor(als_model.item_factors[i]) for i in tqdm.tqdm(range(num_item))}
user_embeddings_dict = {idx_to_user[i]:tf.convert_to_tensor(als_model.user_factors[i]) for i in tqdm.tqdm(range(num_user))}
np.save("item_embeddings_dict.npy", item_embeddings_dict)
np.save("user_embeddings_dict.npy", user_embeddings_dict)
print(len(user_embeddings_dict), len(user_embeddings_dict))

print(torch.cuda.is_available())
'''
item_embeddings_dict = np.load("item_embeddings_dict.npy", allow_pickle=True).item()
user_embeddings_dict = np.load("user_embeddings_dict.npy", allow_pickle=True).item()



''' 
    3. State Representation Models-> DDR-ave
    1) input
    # dataloader에서 나온 return들
    # user_idb : 해당 user의 id 
    # itemb : 유저가 rating 한 item id 10개(tensor))
    # memory :  유저가 rating 한 item들 list 크기는 유저 * 10(item)  
    idx : user_list에서 user의 index
    
    2) output
    state : #state tensor shape [3,100]
'''


def drrave_state_rep(userid_b, memory, idx):
    user_num = idx
    H = []  # item embeddings
    user_n_items = memory
    user_embeddings = torch.Tensor(np.array(user_embeddings_dict[int(userid_b[0])]), ).unsqueeze(0)

    for item in user_n_items:
        H.append(np.array(item_embeddings_dict[int(item)]))
    # avg_layer = nn.AvgPool1d(1)  # pooling layer 사용
    weighted_avg_layer = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1)
    item_embeddings = weighted_avg_layer(torch.Tensor(H, ).unsqueeze(0)).permute(0, 2, 1).squeeze(0)

    state = torch.cat([user_embeddings, user_embeddings * item_embeddings.T, item_embeddings.T])

    return state  # state tensor shape [3,100]


''' 
    4.  Actor Critic model
'''


class Actor(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Actor, self).__init__()

        self.drop_layer = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        x = F.relu(self.linear1(state.to(device)))
        x = self.drop_layer(x)
        x = F.relu(self.linear2(x))
        x = self.drop_layer(x)
        x = self.linear3(x)  # in case embeds are standard scaled / wiped using PCA whitening
        return x  # state = self.state_rep(state)


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Critic, self).__init__()

        self.drop_layer = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(input_dim + output_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state.to(device), action.to(device)], 1)
        x = F.relu(self.linear1(x))
        x = self.drop_layer(x)
        x = F.relu(self.linear2(x))
        x = self.drop_layer(x)
        x = self.linear3(x)
        return x


"""
    5. PER buffer
    1) SegmentTree
    2) ReplayBuffer- NaiveReplayBuffer, PrioritizedReplayBuffer
"""


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        return super(MinSegmentTree, self).reduce(start, end)


# PER ReplayBuffer

# Naive ReplayBuffer
class ReplayBuffer:
    def __init__(self,
                 buffer_size: ('int: total size of the Replay Buffer'),
                 input_dim: ('int: a dimension of input data'),
                 action_dim: ('int: a dimension of action'),
                 batch_size: ('int: a batch size when updating')):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.save_count, self.current_size = 0, 0

        self.state_buffer = np.ones((buffer_size, input_dim), dtype=np.float32)
        self.action_buffer = np.ones((buffer_size, action_dim), dtype=np.float32)
        self.reward_buffer = np.ones(buffer_size, dtype=np.float32)
        self.next_state_buffer = np.ones((buffer_size, input_dim), dtype=np.float32)
        self.done_buffer = np.ones(buffer_size, dtype=np.int8)

    def store(self,
              state: np.float32,
              action: np.float32,
              reward: np.float32,
              next_state: np.float32,
              done: np.int8):
        self.state_buffer[self.save_count] = state
        self.action_buffer[self.save_count] = action
        self.reward_buffer[self.save_count] = reward
        self.next_state_buffer[self.save_count] = next_state
        self.done_buffer[self.save_count] = done

        self.save_count = (self.save_count + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def batch_load(self):
        indices = np.random.randint(self.current_size, size=self.batch_size)
        return dict(
            states=self.state_buffer[indices],
            actions=self.action_buffer[indices],
            rewards=self.reward_buffer[indices],
            next_states=self.next_state_buffer[indices],
            dones=self.done_buffer[indices])


class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self, buffer_size, input_dim, action_dim, batch_size, alpha):

        super(PrioritizedReplayBuffer, self).__init__(buffer_size, input_dim, action_dim, batch_size)

        # For PER. Parameter settings.
        self.max_priority, self.tree_idx = 1.0, 0
        self.alpha = alpha

        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self,
              state: np.float32,
              action: np.float32,
              reward: np.float32,
              next_state: np.ndarray,
              done: np.int8):

        super().store(state, action, reward, next_state, done)

        self.sum_tree[self.tree_idx] = self.max_priority ** self.alpha
        self.min_tree[self.tree_idx] = self.max_priority ** self.alpha
        self.tree_idx = (self.tree_idx + 1) % self.buffer_size

    def batch_load(self, beta):

        indices, p_total = self._sample_indices_with_priority()
        weights = self._cal_weight(indices, p_total, self.current_size, beta)
        return dict(
            states=self.state_buffer[indices],
            actions=self.action_buffer[indices],
            rewards=self.reward_buffer[indices],
            next_states=self.next_state_buffer[indices],
            dones=self.done_buffer[indices],
            weights=weights,
            indices=indices)

    def update_priorities(self, indices, priorities):

        for idx, priority in zip(indices, priorities.flatten()):
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_indices_with_priority(self):

        p_total = self.sum_tree.sum()
        segment = p_total / self.batch_size
        segment_list = [i * segment for i in range(self.batch_size)]
        samples = [np.random.uniform(a, a + segment) for a in segment_list]
        indices = [self.sum_tree.find_prefixsum_idx(sample) for sample in samples]

        return indices, p_total

    def _cal_weight(self, indices, p_total, N, beta):

        p_min = self.min_tree.min() / p_total
        max_weight = (p_min * N) ** (-beta)

        p_samples = np.array([self.sum_tree[idx] for idx in indices]) / p_total
        weights = (p_samples * N) ** (-beta) / max_weight
        return weights


# initializing for drrave state representation
p_loss = []
v_loss = []

value_net = Critic(300, 100, 256).to(device)
policy_net = Actor(300, 100, 256).to(device)

# value_net.load_state_dict(torch.load('./trained_value_diff_drop_600000.pt'))
# policy_net.load_state_dict(torch.load('./trained_policy_diff_drop_600000.pt'))

target_value_net = Critic(300, 100, 256).to(device)
target_policy_net = Actor(300, 100, 256).to(device)

target_policy_net.eval()
target_value_net.eval()

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)

value_criterion = nn.MSELoss()
value_optimizer = AdaBelief(value_net.parameters(), lr=1e-4, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True,
                            rectify=True)
policy_optimizer = AdaBelief(policy_net.parameters(), lr=1e-4, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True,
                             rectify=True)


def ddpg_update(batch_size=32,
                gamma=0.6,
                min_value=-np.inf,
                max_value=np.inf,
                soft_tau=1e-2,
                beta=0.4):
    batch = replay_buffer.batch_load(beta)
    weights = torch.FloatTensor(batch['weights'].reshape(-1, 1)).to(device)  # device 는 GPU 번호
    states = torch.FloatTensor(batch['states']).to(device)
    next_states = torch.FloatTensor(batch['next_states']).to(device)
    actions = torch.FloatTensor(batch['actions']).to(device)
    rewards = torch.FloatTensor(batch['rewards'].reshape(-1, 1)).to(device)
    dones = torch.FloatTensor(batch['dones'].reshape(-1, 1)).to(device)

    # policy loss
    policy_loss = -(weights * value_net(states, policy_net(states))).mean()
    p_loss.append(policy_loss.detach().item())

    # value loss
    # print(states.shape)
    # print(actions.shape)
    value = value_net(states, actions)
    next_actions = target_policy_net(next_states)
    mask = 1 - dones
    expected_value = (rewards + gamma * mask * target_value_net(next_states, next_actions.detach())).to(device)
    expected_value = torch.clamp(expected_value, min_value, max_value)
    sample_wise_loss = F.smooth_l1_loss(value, expected_value.detach(), reduction="none")

    value_loss = (weights * sample_wise_loss).mean()
    v_loss.append(value_loss.detach().item())

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    # For PER: update priorities of the samples.
    epsilon_for_priority = 1e-8
    sample_wise_loss = sample_wise_loss.detach().cpu().numpy()
    batch_priorities = sample_wise_loss + epsilon_for_priority
    replay_buffer.update_priorities(batch['indices'], batch_priorities)

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)



class OfflineEnv2:

    def __init__(self, dataloader, users_dict, item_embeddings_dict, df_cat_title):
        self.dataloader = iter(dataloader)
        self.users_dict = users_dict
        self.df_cat_title = df_cat_title

        self.data = next(self.dataloader)  # {'item':items,'rating':ratings,'size':size,'userid':user_id,'idx':idx}
        self.user_history = self.users_dict[int(self.data['userid'])]

        self.item_embedding = torch.Tensor(
            [np.array(item_embeddings_dict[item]) for item in users_dict[int(self.data['userid'][0])]['item']])
        self.items = self.item_embedding.T.unsqueeze(0)
        self.item_embedding_big = torch.Tensor([np.array(item_embeddings_dict[item]) for item in item_embeddings_dict])
        self.items_big = self.item_embedding_big.T.unsqueeze(0)

        self.memory = [item[0] for item in self.data['item']]
        self.done = 0

        self.related_books = self.generate_related_books()
        self.state_list = self.related_books[:9]  # state을 담아두는 lists 초반엔  기존 state 10개만 담겨 있다

        self.viewed_pos_books = []
        self.next_book_ix = 10

    def generate_related_books(self):
        related_book = []
        items = self.user_history['item']
        ratings = self.user_history['rating']

        for item, rating in zip(items, ratings):
            if rating > 3:
                related_book.append(item)

        return related_book

    def reset(self):
        self.data = next(self.dataloader)
        self.memory = [item[0] for item in self.data['item']]
        self.user_history = self.users_dict[int(self.data['userid'])]
        self.done = 0
        self.item_embedding = torch.Tensor(
            [np.array(item_embeddings_dict[item]) for item in self.users_dict[int(self.data['userid'][0])]['item']])
        self.items = self.item_embedding.T.unsqueeze(0)
        self.related_books = self.generate_related_books()
        self.viewed_pos_books = []
        self.next_book_ix = 10
        self.state_list = self.related_books[:9]

    def update_memory(self, action):
        self.memory = list(self.memory[1:]) + [self.user_history['item'][action]]

    def step(self, action):

        rating = int(self.user_history["rating"][action])

        if rating >= 4:

            book_ix = self.related_books.index(
                self.user_history["item"][action])  # 추천된 책의 4,5점 모여있는 책list(tiemstamp순)에서의 index

            ix_dist = book_ix - self.next_book_ix  # book_ix-10이란 말=> 10개를 추천하려 함

            if ix_dist <= len(self.user_history['rating']) / 10 and ix_dist > 0:
                # 추천 횟수로 나눠주자-> 빠르게 끝나는 것 방지!
                reward = (int(rating) - 3) / 2 * (1 / np.log2(ix_dist + 2))  # rating(0.5~1)*(1)
                #                 self.related_books = self.related_books[book_ix+1]
                self.next_book_ix = book_ix + 1

                self.state_list.append(self.user_history["item"][action])  # bodundary안에만 넣는다, 거기 안에 들어 간 애들만 다시 추천 못함

                self.update_memory(action)
                self.viewed_pos_books.append(action)

            else:
                reward = (int(rating) - 3) / 2 * (1 / np.log2(abs(ix_dist) + 2))
                self.viewed_pos_books.append(action)
        else:
            category_matches = any(
                self.df_cat_title.loc[item, 'category'] == self.df_cat_title.loc[self.state_list[-1], 'category'] for
                item in self.user_history['item'])
            if category_matches:
                reward = 0
            else:
                reward = -1

        # done = 1이 되는 상황 : state의 마지막 순서가 마지막 추천할 아이템이되면 끝내라~!
        if self.related_books[-1] == self.state_list[-1]:
            self.done = 1

        # related_books 마지막까지 갈 수 있는 코드 필요
        return self.memory, reward, self.done

def get_action(state, action_emb, userid_b, items, state_list):
    action_emb = torch.reshape(action_emb, [1, 100]).unsqueeze(0).to(device)
    m = torch.bmm(action_emb, items).squeeze(0)  # torch.bmm : batch 행렬 곱연산
    _, indices = torch.sort(m, descending=True)

    index_list = list(indices[0])

    for i in index_list:
        if users_dict[int(userid_b[0])]["item"][i] not in state_list:  # 기존 state + state가 된 책이 아닌 책들(다시 추천 가능)
            return int(i)


class OUNoise(object):
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.4, min_sigma=0.4, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return torch.tensor([action + ou_state]).float().cpu()


users = dict(tuple(df.groupby("user")))
users_dict = defaultdict(dict)
users_id_list = set()

"""
유저수(user_num) 설정
"""
user_num = 100
for user_id in users:
    rating_freq = Counter(users[user_id]["rating"].values)
    if rating_freq[4]+rating_freq[5]<10 :
        continue
    else:
        users_id_list.add(int(user_id))
        users_dict[user_id]["item"] = users[user_id]["item"].values
        users_dict[user_id]["rating"] = users[user_id]["rating"].values
        user_num -= 1
    if user_num == 0:
        break
print(user_num)


"""
train, test set split
"""
users_id_list = np.array(list(users_id_list))
print(users_id_list)

# choosing default train_test_split of 25%
train_users, test_users = train_test_split(users_id_list, train_size=0.6)
# print(train_users[:2])
# train_users = users_id_list

print(train_users)
print(test_users)

from torch.utils.data import Dataset


class UserDataset(Dataset):
    def __init__(self, users_list, users_dict):
        self.users_list = users_list
        self.users_dict = users_dict

    def __len__(self):
        return len(self.users_list)

    def __getitem__(self, idx):
        user_id = self.users_list[idx]
        items = [('1',)] * 10
        ratings = [('0',)] * 10
        j = 0
        for i, rate in enumerate(self.users_dict[user_id]["rating"]):
            if int(rate) > 3 and j < 10:
                items[j] = self.users_dict[user_id]["item"][i]
                ratings[j] = self.users_dict[user_id]["rating"][i]
                j += 1

        # item = list(self.users_dict[user_id]["item"][:])
        # rating = list(self.users_dict[user_id]["rating"][:])
        size = len(items)

        return {'item': items, 'rating': ratings, 'size': size, 'userid': user_id, 'idx': idx}



train_users_dataset = UserDataset(train_users, users_dict)
test_users_dataset = UserDataset(test_users, users_dict)
whole_users_dataset = UserDataset(users_id_list, users_dict)

train_dataloader = DataLoader(train_users_dataset, batch_size=1)
test_dataloader = DataLoader(test_users_dataset, batch_size=1)
whole_dataloader = DataLoader(whole_users_dataset, batch_size=1)

train_num = len(train_dataloader)
print(train_num)

test_num = len(test_dataloader)
print(test_num)

ou_noise = OUNoise(100, decay_period=10)
buffer_size = 2000
input_dim = 300
action_dim = 100
batch_size = 16
alpha = 0.6
replay_buffer = PrioritizedReplayBuffer(buffer_size, input_dim, action_dim, batch_size, alpha)
memory = np.ones((train_num, 10)) * -1
params = {
    'ou_noise': False
}

accum_rewards = []
mean_rewards = []
beta = 0.4
beta_annealing = (1 - beta) / train_num

iteration = 500
ep = 0
best_score = -11

for i in tqdm.tqdm(range(iteration)):
    env = OfflineEnv2(train_dataloader, users_dict, item_embeddings_dict, df_cat_title)

    for episode in range(train_num):
        if i == 0:
            beta_annealed = min(1, beta + beta_annealing * episode)
        ep = ep + 1
        ep_reward = 0
        batch_size = 8

        item_b, rating_b, size_b, userid_b, idx_b = env.data['item'], env.data['rating'], env.data['size'], env.data['userid'], env.data['idx']
        # print(env.data)
        # print(userid_b)
        memory = env.memory

        state = drrave_state_rep(userid_b, memory, idx_b)
        items = env.items.to(device)

        state_list = env.state_list

        done = 0
        user_len = len(env.user_history['item'])

        iter_num = 0
        for j in range(10):

            if done == 0:
                state_rep = torch.reshape(state, [-1])
                action_emb = policy_net(state_rep)

                # params['ou_noise']:
                #   action_emb = ou_noise.get_action(action_emb.detach().cpu().numpy()[0], j)
                #  action_emb = action_emb.squeeze(0)
                action = get_action(state, action_emb, userid_b, items, state_list)

                memory, reward, done = env.step(action)

                ep_reward += reward

                next_state = drrave_state_rep(userid_b, memory, idx_b)
                next_state_rep = torch.reshape(next_state, [-1])

                replay_buffer.store(state_rep.detach().cpu().numpy(),
                                    action_emb.detach().cpu().numpy(),
                                    reward, next_state_rep.detach().cpu().numpy(), done)
                if replay_buffer.current_size > batch_size:
                    ddpg_update(batch_size=batch_size, beta=beta_annealed)

                state = next_state
                iter_num += 1


            else:
                break

        accum_rewards.append(ep_reward / iter_num)
        if episode < train_num - 1: env.reset()

    iter_reward_mean = np.mean(accum_rewards[(train_num) * i:(train_num) * (i + 1)])
    # reward기준으로 전 iteration보다 높으면 저장하게 해라!
    if iter_reward_mean > best_score:
        #         torch.save(value_net.state_dict(), f'./model/trained_value_it_{i}_split_10_buffer_10k_batch_8_partial.pt')
        #         torch.save(policy_net.state_dict(), f'./model/trained_policy_it_{i}_split_10_buffer_10k_batch_8_partial.pt')
        best_score = iter_reward_mean


print('iteration : ', i, 'mean reward', np.mean(accum_rewards))
plt.figure(facecolor='w', figsize=(30, 8))
plt.plot(accum_rewards)
accum_rewards = np.array(accum_rewards)
print(accum_rewards)

plt.figure(facecolor='w', figsize=(30, 8))
plt.plot(accum_rewards[:2000])
plt.title("first 2000 points")
plt.figure(facecolor='w', figsize=(30, 8))
plt.plot(accum_rewards[-2000:])
plt.title("last 2000 points")


def DCG(label_list):
    dcgsum = 0
    for i in range(len(label_list)):  # len(label_list)
        dcg = (2**label_list[i] - 1)/np.log2(i+2)
        dcgsum += dcg
    return dcgsum


def NDCG(label_list):
    dcg = DCG(label_list[0:len(label_list)])
    ideal_list = sorted(label_list, reverse=True) # reward 기준으로 정렬 ?
    ideal_dcg = DCG(ideal_list[0:len(label_list)])
    if ideal_dcg == 0:
        return 0
    return dcg/ideal_dcg


def get_action_prediction_topk(state, action_emb, userid_b, items, test_pred,related_books,k):
    action_emb = torch.reshape(action_emb,[1,100]).unsqueeze(0).to(device)
    m = torch.bmm(action_emb,items).squeeze(0)  #torch.bmm : batch 행렬 곱연산
    _, indices = torch.sort(m, descending=True)
    index_list = list(indices[0])
    rec_num = 0
    precision_num = 0
    precision_num_topk = 0
    rec_list = []
    rel_list = []
    for i in index_list:
        if users_dict[int(userid_b[0])]["item"][i] not in test_pred:
            rec_list.append(users_dict[int(userid_b[0])]["item"][i])
            rel_list.append(users_dict[int(userid_b[0])]["rating"][i])
            if users_dict[int(userid_b[0])]["rating"][i] >=4:
                precision_num_topk += 1
            rec_num += 1
        if rec_num == k:
            break
    for rec in rec_list:
        if rec in related_books[10:15]:
            precision_num += 1
    return rec_list , rel_list,precision_num, precision_num_topk


def MoreInfo(state_dict, action, answer, user_id):
    #     value = pred_dict[user_id]
    state_item_info = []
    action_info = []
    answer_info = []

    for j in state_dict:  # state에 대한 category 출력
        state_item_info.append(df_cat_title.loc[j][1])
    for k in action:  # action에 대한 category 출력
        action_info.append(df_cat_title.loc[k][1])
    for i in answer:
        answer_info.append(df_cat_title.loc[i][1])

    return state_item_info, action_info, answer_info


def Plot(state, action):
    pass

env = OfflineEnv2(test_dataloader, users_dict, item_embeddings_dict, df_cat_title)
precision = 0
eval_user_num = 4
test_state_dict = dict()

actions = []
users = []
ndcg_all = 0
answer_items = []
topk = 5
precision_all = 0
precision_num_topk_all = 0
dcg_all = 0
for user_idx in range(eval_user_num):  # session 돌리기 : timestamps 내에서 items들
    ep_reward = 0
    item_b, rating_b, size_b, userid_b, idx_b = env.data['item'], env.data['rating'], env.data['size'], env.data[
        'userid'], env.data['idx']
    memory = env.memory
    state = drrave_state_rep(userid_b, memory, idx_b)
    items = env.items.to(device)
    count = 0
    test_first_10 = list([item.item() for item in item_b])
    done = 0
    ndcg = 0
    accum_action = []
    answer_items.append([item.item() for item in item_b])
    users.append(env.data['userid'].item())
    test_state_dict[int(userid_b)] = test_first_10

    state_rep = torch.reshape(state, [-1])
    action_emb = policy_net(state_rep)  # policy_net = actor : items들의 선호도 (rating)
    action_emb = action_emb.squeeze(0)

    if len(env.related_books) >= 15:
        action, rel_list, precision_num, precision_num_topk = get_action_prediction_topk(state, action_emb, userid_b,
                                                                                         items, test_first_10,
                                                                                         env.related_books, topk)
        dcg_user = DCG(rel_list)
        ndcg_5 = NDCG(rel_list)
        precision_user = precision_num / topk
        precision_num_topk_user = precision_num_topk / topk
        ndcg_all += ndcg_5
        dcg_all += dcg_user
        precision_all += precision_user
        precision_num_topk_all += precision_num_topk_user
        answer = env.related_books[10:15]
        state_item_info, action_info, answer_info = MoreInfo(test_first_10, action, answer, userid_b)

        print('User : ', int(userid_b))
        print('User state category: ', state_item_info)
        print('Action : ', action, '\nAnswer :', answer)
        print('Action category: ', action_info, '\nAnswer category: ', answer_info)
        print('rel_list :', rel_list, 'ndcg_5 :', ndcg_5, 'precision_user :', precision_user)
        print('precision_num_topk :', precision_num_topk_user)
        print('DCG_topk :', dcg_user)
        print('-' * 30)
    try:
        env.reset()
    except:
        pass


print('Overall DCG : ', dcg_all / eval_user_num)
print('Overall NDCG : ', ndcg_all / eval_user_num)
print('Overall Precision : ', precision_all / eval_user_num)
print('Overall Precision topk : ', precision_num_topk_all / eval_user_num)