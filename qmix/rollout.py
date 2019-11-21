import numpy as np
import random
import time


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

    def generate_episode(self, epoch, epsilon, blue_model, plan_id, evaluate=False):
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        group_reward = np.array([0.0 for i in range(self.env.our_num)])
        blue_model.reset(plan_id)
        t1 = time.time()
        self.agents.policy.init_hidden(1)  # 初始化hidden_state
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        ship_positions = []
        for i in range(self.env.our_num):
            if self.env.groups[i].group_entity[0].equipment_id == "02010001":               
                avail_action_ship = self.env.get_avail_agent_actions(i)
                act_i_clipped = self.agents.choose_action(self.env.init_obs[i], last_action[i], i, avail_action_ship,
                                                       epsilon, evaluate)
                ship_positions.append({'name_id':self.env.groups[i].group_entity[0].name_id,'position':act_i_clipped})
        self.env.train_set_ship_position(ship_positions)

        self.env.reset()
        t2 = time.time()
        reset_time = t2-t1
        print('sim rum ID:', self.env.sim_run_id, 'epsidoe:', epoch)
        terminated = False
        ep_step = 0
        control_step = 0
        episode_reward = 0
        red_hit_info = []
        blue_hit_info = []
        
        
        while not terminated:
            # time.sleep(0.2)
            # if (step+1) % 5 == 0:
                # print("Generate {} step ".format((step+1)))
            self.env.step_control()
            step_cnt = int(self.env.get_step_cnt())
            self.env.pause_enginee()
            # print("step_cnt: ", step_cnt)
            blue_model.one_step(step_cnt)

            #print(ep_step, step_cnt)
            if ep_step % self.args.step_control == 0:
                # print("aaa",ep_step)
                control_step += 1
                obs = self.env.get_obs()
                state = self.env.get_state()
                actions, avail_actions, actions_onehot = [], [], []
                for agent_id in range(self.n_agents):
                    avail_action = self.env.get_avail_agent_actions(agent_id)
                    action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action,
                                                       epsilon, evaluate)
                    # print("actions: ", action)
                    # 生成对应动作的0 1向量
                    action_onehot = np.zeros(self.args.n_actions)  # dyh
                    action_onehot[action] = 1
                    # action_area = 0
                    # if position != -2:
                    #     action_area = position + (action // 3 - 1) * 6 + (action % 3 - 1)
                    # action_area = 8
                    actions.append(action)
                    actions_onehot.append(action_onehot)
                    avail_actions.append(avail_action)
                    last_action[agent_id] = action_onehot
                    # print("agent ", agent_id, position, action, action_area)
                # dyh reward, terminated, _ = self.env.step(actions)
                # print("actions: ", actions)
                step_reward, dones, info = self.env.step(actions,step_cnt)
                terminated = dones[0]
                reward = info[4]
                red_hit_info = info[2]
                blue_hit_info = info[3]
            else:
                step_reward, dones, info = self.env.step([],step_cnt)
                terminated = dones[0]
                reward = info[4]
                red_hit_info = info[2]
                blue_hit_info = info[3]
            group_reward += step_reward
            self.env.unpause_enginee()

            # print("step_cnt: ", step_cnt)
            if step_cnt > self.args.sim_time - 1:
                terminated = True

            if ep_step % self.args.step_control == 0:
                o.append(obs)
                s.append(state)
                # 和环境交互的actions需要是一个list，里面就装着代表每个agent动作的整数
                # buffer里存的action，每个agent的动作都需要是一个1维向量
                u.append(np.reshape(actions, [self.n_agents, 1]))
                u_onehot.append(actions_onehot)
                avail_u.append(avail_actions)
                r.append([reward])
                terminate.append([terminated])
                padded.append([0.])
            episode_reward += reward
            ep_step += 1
        t3=time.time()
        run_time = t3-t2
            # if terminated:
            #     time.sleep(1)
        # 处理最后一个obs
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # 最后一个obs需要单独计算一下avail_action，到时候需要计算target_q
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # 返回的episode必须长度都是self.episode_limit，所以不够的话进行填充
        for i in range(control_step, self.episode_limit):  # 没有的字段用0填充，并且padded为1
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        '''
        (o[n], u[n], r[n], o_next[n], avail_u[n], u_onehot[n])组成第n条经验，各项维度都为(episode数，transition数，n_agents, 自己的具体维度)
         因为avail_u表示当前经验的obs可执行的动作，但是计算target_q的时候，需要obs_net及其可执行动作，
        '''
        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        return episode, episode_reward, info[0],ep_step,control_step,reset_time,run_time,group_reward,red_hit_info,blue_hit_info
        # 因为buffer里存的是四维的，这里得到的episode只有三维，即transition、agent、shape三个维度，
        # 还差一个episode维度，所以给它加一维
