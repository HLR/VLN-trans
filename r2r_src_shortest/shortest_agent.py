# R2R-EnvDrop, 2019, haotan@cs.unc.edu
# Modified in Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

import json
import numpy as np
import random
from tqdm import tqdm


class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj['path']
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break



class ShortestCollectAgent(BaseAgent):
    ''' An agent that always takes the shortest path to goal. '''
    def __init__(self, env, results_path, max_episode_len, name=""):
        super(ShortestCollectAgent, self).__init__(env, results_path)
        self.episode_len = max_episode_len
        self.name = name


    def collect(self):
        idx = 0
        total_traj = len(self.env.data)
        data = list()
        while len(data) < total_traj:
            traj = self.rollout()
            data.extend(traj)
            print(len(data))
        print("you collected %d shortest paths" % (len(data)))
        new_data = []
        new_data = [item for item in data if item not in new_data]

        file_name = "shortest_{}.json".format(self.name)
        with open(self.results_path + file_name, 'w+') as f:
            json.dump(new_data, f)


    def rollout(self, beam_size=1):
        self.env.reset()
        obs = np.array(self.env._get_obs())
           
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['scan'], ob['viewpoint'], ob['viewIndex'],ob['heading'], ob['elevation'])],
            'teacher_actions':[],
            'teacher_action_emd':[],
            'instr_encoding':ob['instr_encoding']
        } for ob in obs]
        ended = np.array([False] * len(obs))
        #while True:
        for t in range(self.episode_len):
            actions = [ob['teacher'] for ob in obs]
            for i,a in enumerate(actions):
                if not ended[i]:
                    traj[i]['teacher_actions'].append(a)
                    if a == 0:
                        traj[i]['teacher_action_emd'].append((-1,90,90))
                    else:
                        traj[i]['teacher_action_emd'].append((obs[i]['adj_loc_list'][a]['absViewIndex'], obs[i]['adj_loc_list'][a]['rel_heading'],obs[i]['adj_loc_list'][a]['rel_elevation']))
            
            self.env.step(actions, obs)
            obs = self.env._get_obs()
            for i,a in enumerate(actions):
                if a == (0, 0, 0) or a == 0:
                    ended[i] = True
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['scan'], ob['viewpoint'], ob['viewIndex'],ob['heading'], ob['elevation']))
            if ended.all():
                break
        return traj
    

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''

        for iter in tqdm(range(1, n_iters + 1)):
            self.rollout()

