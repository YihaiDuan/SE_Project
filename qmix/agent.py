import numpy as np
import torch
from algorithms.qmix.qmix import QMIX
from torch.distributions import Categorical


class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.policy = QMIX(args)
        self.args = args

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        inputs = obs.copy()
        # print("avail_actions", avail_actions)
        avail_actions_ind = np.nonzero(avail_actions)[0]  # 可执行动作对应的index
        # 传入的agent_num是一个整数，代表第几个agent，现在要把他变成一个onehot向量
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))  # obs是数组，不能append
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]
        # 转化成Tensor,inputs的维度是(42,)，要转化成(1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn.forward(inputs, hidden_state)
        # print("q_value: ", q_value)
        q_value[avail_actions == 0.0] = - float("inf")  # 传入的avail_actions参数是一个array
            # print("q_value: ", q_value)
        
        if not evaluate:
            rand = np.random.uniform()
            if rand < epsilon:
                # print("avail_actions: ", avail_actions_ind)
                action = np.random.choice(avail_actions_ind)  # action是一个整数  dyh avail_actions_ind
                # print ("action: ", action)
            else:
                action = torch.argmax(q_value)
        # print(action)
        else:
            action = torch.argmax(q_value)
        return action.item()


    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma在训练时也需要epsilon计算动作的执行概率
        # 每次学习时，各个episode的长度不一样，因此取其中最长的episode作为所有episode的长度
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        # if train_step > 0 and train_step % self.args.save_cycle == 0:
        #     self.policy.save_model(train_step)











