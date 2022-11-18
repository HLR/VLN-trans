''' Agents: stop/random/shortest/seq2seq  '''

import json
import os
import sys
import numpy as np
import random
import time
import pickle

import torch
import torch.nn as nn
import torch.distributions as D
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import copy
from env import debug_beam
from utils import padding_idx, to_contiguous, clip_gradient
from agent_utils import basic_actions, sort_batch, teacher_action, discount_rewards, backchain_inference_states, path_element_from_observation, InferenceState, WorldState, least_common_viewpoint_path

from collections import Counter, defaultdict
import pdb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

testing_settingA=False

# region Simple Agents
class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path, seed=1):
        self.env = env
        self.results_path = results_path
        if seed != 'resume': random.seed(seed)
        self.results = {}
        self.losses = [] # For learning agents

        self.testing_settingA = False

    def write_results(self, dump_file=False):
        if '_' in list(self.results.keys())[0]:
            #if testing_settingA:
            if self.testing_settingA:
                # choose one from three according to prob
                for id in self.results:
                    bestp, best = self.results[id][1], self.results[id]
                    for ii in range(4):
                        temp_id = "%s_%d" % (id[:-2], ii)
                        if temp_id in self.results and self.results[temp_id][1] > bestp:
                            bestp = self.results[temp_id][1]
                            best = self.results[temp_id]
                    self.results[id] = best
                output = [{'instr_id': k, 'trajectory': v[0]} for k, v in self.results.items()]
            else:
                output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        else:
            output = [{'instr_id':'%s_%d' % (k, i), 'trajectory': v} for k,v in self.results.items() for i in range(self.env.traj_n_sents[k])]

        if dump_file:
            with open(self.results_path, 'w') as f:
                json.dump(output, f)
        return output

    def rollout(self, beam_size=1):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, beam_size=1, successors=1, speaker=(None,None,None,None)):
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        #print 'Testing %s' % self.__class__.__name__
        speaker, speaker_weights, speaker_merge, evaluator = speaker
        looped = False
        batch_i, index_count = 0, [Counter() for _ in range(len(speaker_weights))] if speaker else []# for beam search
        while True:
            trajs = self.rollout(beam_size, successors)
            if beam_size > 1 or debug_beam:
                trajs, completed, traversed_list = trajs

            for ti, traj in enumerate(trajs):
                if (beam_size == 1 and debug_beam) or (beam_size>1 and speaker is None):
                    traj = traj[0]
                elif beam_size>1 and (speaker is not None):#use speaker
                    traj = speaker_rank(speaker, speaker_weights, speaker_merge, traj, completed[ti], traversed_list[ti] if traversed_list else None, index_count)
                else:
                    assert (beam_size == 1 and not debug_beam)

                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    #if testing_settingA:
                    if self.testing_settingA:
                        self.results[traj['instr_id']] = (traj['path'], traj['prob'])  # choose one from three according to prob
                    else:
                        self.results[traj['instr_id']] = traj['path']

            if looped:
                break
            if beam_size>1: print('batch',batch_i)
            batch_i+=1
        # if use speaker, find best weight
        if beam_size>1 and (speaker is not None):  # speaker's multiple choices
            best_sr, best_speaker_weight_i = -1, -1
            for spi, speaker_weight in enumerate(speaker_weights):

                if '_' in list(self.results.keys())[0]:
                    output = [{'instr_id': k, 'trajectory': v[spi]} for k, v in self.results.items()]
                else:
                    output = [{'instr_id': '%s_%d' % (k, i), 'trajectory': v[spi]} for k, v in self.results.items() for i in
                              range(self.env.traj_n_sents[k])]

                score_summary, _ = evaluator.score_output(output)
                data_log = defaultdict(list)
                for metric, val in score_summary.items():
                    data_log['%s %s' % (''.join(evaluator.splits), metric)].append(val)
                print(index_count[spi])
                print(speaker_weights[spi], '\n'.join([str((k, round(v[0], 4))) for k, v in sorted(data_log.items())]))

                sr = score_summary['success_rate']
                if sr>best_sr:
                    best_sr, best_speaker_weight_i = sr, spi
            print('best sr:',best_sr,' speaker weight:',speaker_weights[best_speaker_weight_i])
            print('best sr counter', index_count[best_speaker_weight_i])
            self.results = {k: v[best_speaker_weight_i] for k, v in self.results.items()}


def speaker_rank(speaker, speaker_weights, speaker_merge, beam_candidates, this_completed, traversed_lists, index_count):  # todo: this_completed is not sorted!! so not corresponding to beam_candidates
    cand_obs, cand_actions, multi = [], [], isinstance(beam_candidates[0]['instr_encoding'], list)
    cand_instr = [[] for _ in beam_candidates[0]['instr_encoding']] if multi else []  # else should be np.narray
    for candidate in beam_candidates:
        cand_obs.append(candidate['observations'])
        cand_actions.append(candidate['actions'])
        if multi:
            for si, encoding in enumerate(candidate['instr_encoding']):
                cand_instr[si].append(np.trim_zeros(encoding)[:-1])
        else:
            cand_instr.append(np.trim_zeros(candidate['instr_encoding'])[:-1])
    if multi:
        speaker_scored_candidates = [[] for _ in (beam_candidates)]
        for si, sub_cand_instr in enumerate(cand_instr):
            speaker_scored_candidates_si, _ = \
                speaker._score_obs_actions_and_instructions(
                    cand_obs, cand_actions, sub_cand_instr, feedback='teacher')
            for sc_i, sc in enumerate(speaker_scored_candidates_si):
                speaker_scored_candidates[sc_i].append(sc)
    else:
        speaker_scored_candidates, _ = \
            speaker._score_obs_actions_and_instructions(
                cand_obs, cand_actions, cand_instr, feedback='teacher')
        assert len(speaker_scored_candidates) == len(beam_candidates)

    follower_scores = []
    speaker_scores = []

    score_merge = {'mean':np.mean,'max':np.max,'min':np.min}[speaker_merge]

    for i, candidate in enumerate(beam_candidates):  # different to speaker follower, our beam_candidates is not nested, we already got a subset from the outside of this function, so we do not need flatten it before enumerate
        speaker_scored_candidate = speaker_scored_candidates[i]
        if multi:
            assert candidate['instr_id'] == speaker_scored_candidate[0]['instr_id']
            candidate['speaker_score'] = score_merge([s['score'] for s in speaker_scored_candidate])
        else:
            assert candidate['instr_id'] == speaker_scored_candidate['instr_id']
            candidate['speaker_score'] = speaker_scored_candidate['score']

        candidate['follower_score'] = candidate['score']
        del candidate['observations']
        if traversed_lists:# physical_traversal:
            last_traversed = traversed_lists[-1]
            candidate_inf_state = \
                this_completed[i]
            path_from_last_to_next = least_common_viewpoint_path(
                last_traversed, candidate_inf_state)
            assert path_from_last_to_next[0].world_state.viewpointId \
                   == last_traversed.world_state.viewpointId
            assert path_from_last_to_next[-1].world_state.viewpointId \
                   == candidate_inf_state.world_state.viewpointId

            inf_traj = (traversed_lists +
                        path_from_last_to_next[1:])
            physical_trajectory = [
                path_element_from_observation(inf_state.observation)
                for inf_state in inf_traj]
            # make sure the viewpointIds match
            assert (physical_trajectory[-1][0] ==
                    candidate['path'][-1][0])
            candidate['path'] = physical_trajectory

        follower_scores.append(candidate['follower_score'])
        speaker_scores.append(candidate['speaker_score'])


    speaker_std = np.std(speaker_scores)
    follower_std = np.std(follower_scores)

    instr_id = beam_candidates[0]['instr_id']
    result_path = []

    for spi, speaker_weight in enumerate(speaker_weights):
        speaker_scaled_weight = float(speaker_weight) / speaker_std
        follower_scaled_weight = (1 - float(speaker_weight)) / follower_std

        best_ix, best_cand = max(
            enumerate(beam_candidates),
            key=lambda tp: (
                    tp[1]['speaker_score'] * speaker_scaled_weight +
                    tp[1]['follower_score'] * follower_scaled_weight))
        result_path.append(best_cand['path'])
        index_count[spi][best_ix] += 1
    return  {'instr_id': instr_id, 'path': result_path}

class StopAgent(BaseAgent):
    ''' An agent that doesn't move! '''

    def rollout(self, beam_size=1):
        world_states = self.env.reset()
        obs = np.array(self.env._get_obs(world_states))
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        return traj


class RandomAgent(BaseAgent):
    ''' An agent that picks a random direction then tries to go straight for
        five viewpoint steps and then stops. '''
    def __init__(self, env, results_path):
        super(RandomAgent, self).__init__(env, results_path)
        random.seed(1)

    def rollout(self, beam_size=1):
        world_states = self.env.reset()
        obs = self.env._get_obs(world_states)
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        self.steps = random.sample(list(range(-11,1)), len(obs))
        ended = [False] * len(obs)
        for t in range(30):
            actions = []
            for i,ob in enumerate(obs):
                if self.steps[i] >= 5:
                    actions.append((0, 0, 0)) # do nothing, i.e. end
                    ended[i] = True
                elif self.steps[i] < 0:
                    actions.append((0, 1, 0)) # turn right (direction choosing)
                    self.steps[i] += 1
                elif len(ob['navigableLocations']) > 1:
                    actions.append((1, 0, 0)) # go forward
                    self.steps[i] += 1
                else:
                    actions.append((0, 1, 0)) # turn right until we can go forward
            obs = self.env._get_obs(self.env.step(actions))
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
        return traj


class ShortestAgent(BaseAgent):
    ''' An agent that always takes the shortest path to goal. '''

    def rollout(self, beam_size=1):
        world_states = self.env.reset()
        obs = np.array(self.env._get_obs(world_states))
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]
        ended = np.array([False] * len(obs))
        while True:
            actions = [ob['teacher'] for ob in obs]
            obs = self.env._get_obs(self.env.step(actions))
            for i,a in enumerate(actions):
                if a == (0, 0, 0):
                    ended[i] = True
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))
            if ended.all():
                break
        return traj


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
        print("you collected %d shortest paths" % (len(data)))
        file_name = "/shortest_{}.json".format(self.name)
        with open(self.results_path + file_name, 'w+') as f:
            json.dump(data, f)


    def rollout(self, beam_size=1):
        world_states = self.env.reset(False)
        obs = np.array(self.env._get_obs(world_states))
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['scan'], ob['viewpoint'], ob['viewIndex'],ob['heading'], ob['elevation'])],
            'teacher_actions':[],
            'teacher_action_emd':[],
            'instr_encoding':ob['instr_encoding'].tolist()
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

            obs = self.env._get_obs(self.env.step(actions, obs))
            for i,a in enumerate(actions):
                if a == (0, 0, 0) or a == 0:
                    ended[i] = True
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['scan'], ob['viewpoint'], ob['viewIndex'],ob['heading'], ob['elevation']))
            if ended.all():
                break
        return traj
# endregion

class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''
    model_actions, env_actions = basic_actions()
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, encoder, decoder, seed, aux_ratio, decoder_init,
                 params=None, monotonic=False, episode_len=20, state_factored=False, accu_n_iters = 0):  # , subgoal
        super(Seq2SeqAgent, self).__init__(env, results_path, seed=seed)
        self.encoder, self.decoder = encoder, decoder  # encoder2 is only for self_critic
        self.encoder2, self.decoder2 = None, None
        self.monotonic = monotonic
        if self.monotonic:
            self.copy_seq2seq()
        self.episode_len = episode_len
        self.losses = []
        self.losses_ctrl_f = [] # For learning auxiliary tasks
        self.aux_ratio = aux_ratio
        self.decoder_init = decoder_init

        self.clip_gradient = params['clip_gradient']
        self.clip_gradient_norm = params['clip_gradient_norm']
        self.reward_func = params['reward_func']

        self.schedule_ratio = params['schedule_ratio']
        self.temp_alpha = params['temp_alpha']

        self.testing_settingA = params['test_A']

        if self.decoder.action_space == 6:
            self.ignore_index = self.model_actions.index('<ignore>')
        else:
            self.ignore_index = -1
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        if self.decoder.ctrl_feature:
            assert self.decoder.action_space == -1 # currently only implement this
            self.criterion_ctrl_f = nn.MSELoss()  # todo: MSE or ?

        self.state_factored = state_factored
        self.accu_n_iters = accu_n_iters

    @staticmethod
    def n_inputs():
        return len(Seq2SeqAgent.model_actions)

    @staticmethod
    def n_outputs():
        return len(Seq2SeqAgent.model_actions)-2 # Model doesn't output start or ignore

    def _sort_batch(self, obs):
        sorted_tensor, mask, seq_lengths, perm_idx = sort_batch(obs)

        if isinstance(sorted_tensor, list):
            sorted_tensors, masks, seqs_lengths = [], [], []
            for i in range(len(sorted_tensor)):
                sorted_tensors.append(Variable(sorted_tensor[i], requires_grad=False).long().to(device))
                masks.append(mask[i].byte().to(device))
                seqs_lengths.append(seq_lengths[i])
            return sorted_tensors, masks, seqs_lengths, perm_idx

        return Variable(sorted_tensor, requires_grad=False).long().to(device), \
               mask.byte().to(device), \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        #feature_size = obs[0]['feature'].shape[0]
        #features = np.empty((len(obs),feature_size), dtype=np.float32)
        if isinstance(obs[0]['feature'],tuple): # todo?
            features_pano = np.repeat(np.expand_dims(np.zeros_like(obs[0]['feature'][0], dtype=np.float32), 0), len(obs), axis=0)  # jolin
            features = np.repeat(np.expand_dims(np.zeros_like(obs[0]['feature'][1], dtype=np.float32), 0), len(obs), axis=0)  # jolin
            for i,ob in enumerate(obs):
                features_pano[i] = ob['feature'][0]
                features[i] = ob['feature'][1]
            return (Variable(torch.from_numpy(features_pano), requires_grad=False).to(device),
            Variable(torch.from_numpy(features), requires_grad=False).to(device))
        else:
            features = np.repeat(np.expand_dims(np.zeros_like(obs[0]['feature'], dtype=np.float32),0),len(obs),axis=0)  # jolin
            for i,ob in enumerate(obs):
                features[i] = ob['feature']
            return Variable(torch.from_numpy(features), requires_grad=False).to(device)


    def get_next(self, feedback, target, logit):
        if feedback == 'teacher':
            a_t = target  # teacher forcing
        elif feedback == 'argmax':
            _, a_t = logit.max(1)  # student forcing - argmax
            a_t = a_t.detach()
        elif feedback == 'sample':
            probs = F.softmax(logit, dim=1)
            m = D.Categorical(probs)
            a_t = m.sample()  # sampling an action from model
        else:
            sys.exit('Invalid feedback option')
        return a_t

    def _action_variable(self, obs):
        # get the maximum number of actions of all sample in this batch
        max_num_a = -1
        for i, ob in enumerate(obs):
            max_num_a = max(max_num_a, len(ob['adj_loc_list']))

        is_valid = np.zeros((len(obs), max_num_a), np.float32)
        action_embedding_dim = obs[0]['action_embedding'].shape[-1]
        action_embeddings = np.zeros((len(obs), max_num_a, action_embedding_dim), dtype=np.float32)
        for i, ob in enumerate(obs):
            adj_loc_list = ob['adj_loc_list']
            num_a = len(adj_loc_list)
            is_valid[i, 0:num_a] = 1.
            action_embeddings[i, :num_a, :] = ob['action_embedding'] #bug: todo
            #for n_a, adj_dict in enumerate(adj_loc_list):
            #    action_embeddings[i, :num_a, :] = ob['action_embedding']
        return (Variable(torch.from_numpy(action_embeddings), requires_grad=False).to(device),
                Variable(torch.from_numpy(is_valid), requires_grad=False).to(device),
                is_valid)

    def _teacher_action_variable(self, obs):
        # get the maximum number of actions of all sample in this batch
        action_embedding_dim = obs[0]['action_embedding'].shape[-1]
        action_embeddings = np.zeros((len(obs), action_embedding_dim), dtype=np.float32)
        for i, ob in enumerate(obs):
            adj_loc_list = ob['adj_loc_list']
            action_embeddings[i, :] = ob['action_embedding'][ob['teacher']] #bug: todo
            #for n_a, adj_dict in enumerate(adj_loc_list):
            #    action_embeddings[i, :num_a, :] = ob['action_embedding']
        return Variable(torch.from_numpy(action_embeddings), requires_grad=False).to(device)

    def _teacher_action(self, obs, ended):
        a = teacher_action(self.model_actions, self.decoder.action_space, obs, ended, self.ignore_index)
        return Variable(a, requires_grad=False).to(device)

    def _teacher_feature(self, obs, ended):#, max_num_a):
        ''' Extract teacher look ahead auxiliary features into variable. '''
        # todo: 6 action space
        ctrl_features_dim = -1
        for i, ob in enumerate(obs):  # todo: whether include <stop> ?
            # max_num_a = max(max_num_a, len(ob['ctrl_features']))
            if ctrl_features_dim<0 and len(ob['ctrl_features']):
                ctrl_features_dim = ob['ctrl_features'].shape[-1] #[0].shape[-1]
                break

        #is_valid no need to create. already created
        ctrl_features_tensor = np.zeros((len(obs), ctrl_features_dim), dtype=np.float32)
        for i, ob in enumerate(obs):
            if not ended[i]:
                ctrl_features_tensor[i, :] = ob['ctrl_features']
        return Variable(torch.from_numpy(ctrl_features_tensor), requires_grad=False).to(device)

    def rollout(self, beam_size=1, successors=1):
        if beam_size ==1 and not debug_beam:
            if self.encoder.__class__.__name__ in ['BertImgEncoder','MultiVilBertEncoder','BertAddEncoder','MultiVilAddEncoder','MultiAddLoadEncoder', 'HugAddEncoder','MultiHugAddEncoder','VicEncoder','MultiVicEncoder','DicEncoder','MultiDicEncoder']:
                return self.bert_rollout_with_loss()
            elif self.encoder.__class__.__name__ in ['BertLangEncoder']:
                return self.langbert_rollout_with_loss()
            else:
                return self.rollout_with_loss()

        # beam
        with torch.no_grad():
            if self.state_factored:
                beams = self.state_factored_search(beam_size, successors, first_n_ws_key=4)
            else:
                beams = self.beam_search(beam_size)
        return beams

    def state_factored_search(self, completion_size, successor_size, first_n_ws_key=4):
        assert self.decoder.panoramic
        world_states = self.env.reset(sort=True)
        initial_obs = (self.env._get_obs(world_states))
        batch_size = len(world_states)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch([o for ob in initial_obs for o in ob])

        world_states = [[world_state for f, world_state in states] for states in world_states]

        ctx, h_t, c_t, seq_mask = self.encoder(seq, seq_mask, seq_lengths)
        if not self.decoder_init:
            h_t, c_t = torch.zeros_like(h_t), torch.zeros_like(c_t)

        completed = []
        completed_holding = []
        for _ in range(batch_size):
            completed.append({})
            completed_holding.append({})

        state_cache = [
            {ws[0][0:first_n_ws_key]: (InferenceState(prev_inference_state=None,
                                                      world_state=ws[0],
                                                      observation=o[0],
                                                      flat_index=None,
                                                      last_action=-1,
                                                      last_action_embedding=self.decoder.u_begin,
                                                      action_count=0,
                                                      score=0.0, h_t=h_t[i], c_t=c_t[i], last_alpha=None), True)}
            for i, (ws, o) in enumerate(zip(world_states, initial_obs))
        ]

        beams = [[inf_state for world_state, (inf_state, expanded) in sorted(instance_cache.items())]
                 for instance_cache in state_cache] # sorting is a noop here since each instance_cache should only contain one

        last_expanded_list = []
        traversed_lists = []

        for beam in beams:
            assert len(beam)==1
            first_state = beam[0]
            last_expanded_list.append(first_state)
            traversed_lists.append([first_state])

        def update_traversed_lists(new_visited_inf_states):
            assert len(new_visited_inf_states) == len(last_expanded_list)
            assert len(new_visited_inf_states) == len(traversed_lists)

            for instance_index, instance_states in enumerate(new_visited_inf_states):
                last_expanded = last_expanded_list[instance_index]
                # todo: if this passes, shouldn't need traversed_lists
                assert last_expanded.world_state.viewpointId == traversed_lists[instance_index][-1].world_state.viewpointId
                for inf_state in instance_states:
                    path_from_last_to_next = least_common_viewpoint_path(last_expanded, inf_state)
                    # path_from_last should include last_expanded's world state as the first element, so check and drop that
                    assert path_from_last_to_next[0].world_state.viewpointId == last_expanded.world_state.viewpointId
                    assert path_from_last_to_next[-1].world_state.viewpointId == inf_state.world_state.viewpointId
                    traversed_lists[instance_index].extend(path_from_last_to_next[1:])
                    last_expanded = inf_state
                last_expanded_list[instance_index] = last_expanded


        # Do a sequence rollout and calculate the loss
        while any(len(comp) < completion_size for comp in completed):
            beam_indices = []
            u_t_list = []
            h_t_list = []
            c_t_list = []
            flat_obs = []
            for beam_index, beam in enumerate(beams):
                for inf_state in beam:
                    beam_indices.append(beam_index)
                    u_t_list.append(inf_state.last_action_embedding)
                    h_t_list.append(inf_state.h_t.unsqueeze(0))
                    c_t_list.append(inf_state.c_t.unsqueeze(0))
                    flat_obs.append(inf_state.observation)

            u_t_prev = torch.stack(u_t_list, dim=0)
            assert len(u_t_prev.shape) == 2
            # Image features from obs
            # if self.decoder.panoramic:
            f_t_all, f_t = self._feature_variable(flat_obs)


            # Action feature from obs
            # if self.decoder.action_space == 6:
            #     u_t_features, is_valid = np.zeros((batch_size, 1)), None
            # else:
            u_t_features, is_valid, is_valid_numpy = self._action_variable(flat_obs)

            h_t = torch.cat(h_t_list, dim=0)
            c_t = torch.cat(c_t_list, dim=0)

            h_t, c_t, alpha, logit, pred_f = self.decoder(None, u_t_prev, u_t_features, f_t,
                                                          f_t_all, h_t, c_t, [ctx_si[beam_indices] for ctx_si in ctx] if isinstance(ctx, list) else ctx[beam_indices],
                  [seq_mask_si[beam_indices] for seq_mask_si in seq_mask] if isinstance(ctx, list) else seq_mask[beam_indices])

            # Mask outputs of invalid actions
            logit[is_valid == 0] = -float('inf')
            # # Mask outputs where agent can't move forward
            # no_forward_mask = [len(ob['navigableLocations']) <= 1 for ob in flat_obs]

            masked_logit = logit
            log_probs = F.log_softmax(logit, dim=1).data

            # force ending if we've reached the max time steps
            # if t == self.episode_len - 1:
            #     action_scores = log_probs[:,self.end_index].unsqueeze(-1)
            #     action_indices = torch.from_numpy(np.full((log_probs.size()[0], 1), self.end_index))
            # else:
            #_, action_indices = masked_logit.data.topk(min(successor_size, logit.size()[1]), dim=1)
            _, action_indices = masked_logit.data.topk(logit.size()[1], dim=1) # todo: fix this
            action_scores = log_probs.gather(1, action_indices)
            assert action_scores.size() == action_indices.size()

            start_index = 0
            assert len(beams) == len(world_states)
            all_successors = []
            for beam_index, (beam, beam_world_states) in enumerate(zip(beams, world_states)):
                successors = []
                end_index = start_index + len(beam)
                assert len(beam_world_states) == len(beam)
                if beam:
                    for inf_index, (inf_state, world_state, action_score_row) in \
                            enumerate(zip(beam, beam_world_states, log_probs[start_index:end_index])):
                        flat_index = start_index + inf_index
                        for action_index, action_score in enumerate(action_score_row):
                            if is_valid_numpy[flat_index, action_index] == 0:
                                continue
                            successors.append(
                                InferenceState(prev_inference_state=inf_state, world_state=world_state,
                                               observation=flat_obs[flat_index],
                                               flat_index=None,
                                               last_action=action_index,
                                               last_action_embedding=u_t_features[flat_index, action_index].detach(),
                                               action_count=inf_state.action_count + 1,
                                               score=float(inf_state.score + action_score),
                                               h_t=h_t[flat_index], c_t=c_t[flat_index],
                                               last_alpha=[alpha_si[flat_index].data for alpha_si in alpha] if isinstance(alpha, list) else alpha[flat_index].data)
                            )
                start_index = end_index
                successors = sorted(successors, key=lambda t: t.score, reverse=True)
                all_successors.append(successors)

            successor_world_states = [
                [inf_state.world_state for inf_state in successors]
                for successors in all_successors
            ]

            successor_env_actions = [
                [inf_state.last_action for inf_state in successors]
                for successors in all_successors
            ]

            successor_last_obs = [
                [inf_state.observation for inf_state in successors]
                for successors in all_successors
            ]

            successor_world_states = self.env.step(successor_env_actions, successor_last_obs, successor_world_states)
            successor_world_states = [[world_state for f, world_state in states] for states in successor_world_states]

            acc = []
            for ttt in zip(all_successors, successor_world_states):
                mapped = [inf._replace(world_state=ws) for inf, ws in zip(*ttt)]
                acc.append(mapped)
            all_successors = acc

            assert len(all_successors) == len(state_cache)

            new_beams = []

            for beam_index, (successors, instance_cache) in enumerate(zip(all_successors, state_cache)):
                # early stop if we've already built a sizable completion list
                instance_completed = completed[beam_index]
                instance_completed_holding = completed_holding[beam_index]
                if len(instance_completed) >= completion_size:
                    new_beams.append([])
                    continue
                for successor in successors:
                    ws_keys = successor.world_state[0:first_n_ws_key]
                    if successor.last_action == 0 or successor.action_count == self.episode_len:
                        if ws_keys not in instance_completed_holding or instance_completed_holding[ws_keys][
                            0].score < successor.score:
                            instance_completed_holding[ws_keys] = (successor, False)
                    else:
                        if ws_keys not in instance_cache or instance_cache[ws_keys][0].score < successor.score:
                            instance_cache[ws_keys] = (successor, False)

                # third value: did this come from completed_holding?
                uncompleted_to_consider = ((ws_keys, inf_state, False) for (ws_keys, (inf_state, expanded)) in
                                           instance_cache.items() if not expanded)
                completed_to_consider = ((ws_keys, inf_state, True) for (ws_keys, (inf_state, expanded)) in
                                         instance_completed_holding.items() if not expanded)
                import itertools
                import heapq
                to_consider = itertools.chain(uncompleted_to_consider, completed_to_consider)
                ws_keys_and_inf_states = heapq.nlargest(successor_size, to_consider, key=lambda pair: pair[1].score)

                new_beam = []
                for ws_keys, inf_state, is_completed in ws_keys_and_inf_states:
                    if is_completed:
                        assert instance_completed_holding[ws_keys] == (inf_state, False)
                        instance_completed_holding[ws_keys] = (inf_state, True)
                        if ws_keys not in instance_completed or instance_completed[ws_keys].score < inf_state.score:
                            instance_completed[ws_keys] = inf_state
                    else:
                        instance_cache[ws_keys] = (inf_state, True)
                        new_beam.append(inf_state)

                if len(instance_completed) >= completion_size:
                    new_beams.append([])
                else:
                    new_beams.append(new_beam)

            beams = new_beams

            # Early exit if all ended
            if not any(beam for beam in beams):
                break

            world_states = [
                [inf_state.world_state for inf_state in beam]
                for beam in beams
            ]
            successor_obs = np.array(self.env._get_obs(self.env.world_states2feature_states(world_states)))

            acc = []
            for tttt in zip(beams, successor_obs):
                mapped = [inf._replace(observation=o) for inf, o in zip(*tttt)]
                acc.append(mapped)
            beams = acc
            update_traversed_lists(beams)

        completed_list = []
        for this_completed in completed:
            completed_list.append(sorted(this_completed.values(), key=lambda t: t.score, reverse=True)[:completion_size])
        completed_ws = [
            [inf_state.world_state for inf_state in comp_l]
            for comp_l in completed_list
        ]
        completed_obs = np.array(self.env._get_obs(self.env.world_states2feature_states(completed_ws)))
        accu = []
        for ttttt in zip(completed_list, completed_obs):
            mapped = [inf._replace(observation=o) for inf, o in zip(*ttttt)]
            accu.append(mapped)
        completed_list = accu

        update_traversed_lists(completed_list)

        trajs = []
        for this_completed in completed_list:
            assert this_completed
            this_trajs = []
            for inf_state in this_completed:
                path_states, path_observations, path_actions, path_scores, path_attentions = backchain_inference_states(inf_state)
                # this will have messed-up headings for (at least some) starting locations because of
                # discretization, so read from the observations instead
                ## path = [(obs.viewpointId, state.heading, state.elevation)
                ##         for state in path_states]
                trajectory = [path_element_from_observation(ob) for ob in path_observations]
                this_trajs.append({
                    'instr_id': path_observations[0]['instr_id'],
                    'instr_encoding': path_observations[0]['instr_encoding'],
                    'path': trajectory,
                    'observations': path_observations,
                    'actions': path_actions,
                    'score': inf_state.score,
                    'scores': path_scores,
                    'attentions': path_attentions
                })
            trajs.append(this_trajs)
        return trajs, completed_list, traversed_lists


    def beam_search(self, beam_size):
        assert self.decoder.panoramic
        # assert self.env.beam_size >= beam_size
        world_states = self.env.reset(True)  # [(feature, state)]
        obs = np.array(self.env._get_obs(world_states))
        batch_size = len(world_states)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch([o for ob in obs for o in ob])

        world_states = [[world_state for f, world_state in states] for states in world_states]

        ctx, h_t, c_t, seq_mask = self.encoder(seq, seq_mask, seq_lengths)
        if not self.decoder_init:
            h_t, c_t = torch.zeros_like(h_t), torch.zeros_like(c_t)

        completed = []
        for _ in range(batch_size):
            completed.append([])

        beams = [[InferenceState(prev_inference_state=None,
                                 world_state=ws[0],
                                 observation=o[0],
                                 flat_index=i,
                                 last_action=-1,
                                 last_action_embedding=self.decoder.u_begin,
                                 action_count=0,
                                 score=0.0,
                                 h_t=None, c_t=None,
                                 last_alpha=None)]
                 for i, (ws, o) in enumerate(zip(world_states, obs))]
        #
        # batch_size x beam_size

        for t in range(self.episode_len):
            flat_indices = []
            beam_indices = []
            u_t_list = []
            for beam_index, beam in enumerate(beams):
                for inf_state in beam:
                    beam_indices.append(beam_index)
                    flat_indices.append(inf_state.flat_index)
                    u_t_list.append(inf_state.last_action_embedding)

            u_t_prev = torch.stack(u_t_list, dim=0)
            flat_obs = [ob for obs_beam in obs for ob in obs_beam]

            # Image features from obs
            # if self.decoder.panoramic:
            f_t_all, f_t = self._feature_variable(flat_obs)


            # Action feature from obs
            # if self.decoder.action_space == 6:
            #     u_t_features, is_valid = np.zeros((batch_size, 1)), None
            # else:
            u_t_features, is_valid, is_valid_numpy = self._action_variable(flat_obs)

            h_t, c_t, alpha, logit, pred_f = self.decoder(None, u_t_prev, u_t_features,
                                                          f_t, f_t_all, h_t[flat_indices], c_t[flat_indices],
                  [ctx_si[beam_indices] for ctx_si in ctx] if isinstance(ctx, list) else ctx[beam_indices],
                  [seq_mask_si[beam_indices] for seq_mask_si in seq_mask] if isinstance(ctx, list) else seq_mask[beam_indices])
            # Mask outputs where agent can't move forward

            logit[is_valid == 0] = -float('inf')

            masked_logit = logit  # for debug
            log_probs = F.log_softmax(logit, dim=1).data

            _, action_indices = masked_logit.data.topk(min(beam_size, logit.size()[1]), dim=1)
            action_scores = log_probs.gather(1, action_indices)
            assert action_scores.size() == action_indices.size()

            start_index = 0
            new_beams = []
            assert len(beams) == len(world_states)
            all_successors = []
            for beam_index, (beam, beam_world_states, beam_obs) in enumerate(zip(beams, world_states, obs)):
                successors = []
                end_index = start_index + len(beam)
                assert len(beam_obs) == len(beam) and len(beam_world_states) == len(beam)
                if beam:
                    for inf_index, (inf_state, world_state, ob, action_score_row, action_index_row) in \
                            enumerate(zip(beam, beam_world_states, beam_obs, action_scores[start_index:end_index],
                                          action_indices[start_index:end_index])):
                        flat_index = start_index + inf_index
                        for action_score, action_index in zip(action_score_row, action_index_row):
                            if is_valid_numpy[flat_index, action_index] == 0:
                                continue
                            successors.append(
                                InferenceState(prev_inference_state=inf_state,
                                   world_state=world_state,
                                   # will be updated later after successors are pruned
                                   observation=ob,  # will be updated later after successors are pruned
                                   flat_index=flat_index,
                                   last_action=action_index,
                                   last_action_embedding=u_t_features[flat_index, action_index].detach(),
                                   action_count=inf_state.action_count + 1,
                                   score=float(inf_state.score + action_score), h_t=None, c_t=None,
                                   last_alpha=[alpha_si[flat_index].data for alpha_si in alpha] if isinstance(alpha, list) else alpha[flat_index].data)
                            )
                start_index = end_index
                successors = sorted(successors, key=lambda t: t.score, reverse=True)[:beam_size]
                all_successors.append(successors)

            successor_world_states = [
                [inf_state.world_state for inf_state in successors]
                for successors in all_successors
            ]

            successor_env_actions = [
                [inf_state.last_action for inf_state in successors]
                for successors in all_successors
            ]

            successor_last_obs = [
                [inf_state.observation for inf_state in successors]
                for successors in all_successors
            ]

            successor_world_states=self.env.step(successor_env_actions, successor_last_obs, successor_world_states)
            successor_obs = np.array(self.env._get_obs(successor_world_states))
            successor_world_states = [[world_state for f, world_state in states] for states in successor_world_states]

            acc = []
            for ttt in zip(all_successors, successor_world_states, successor_obs):
                mapped = [inf._replace(world_state=ws, observation=o) for inf, ws, o in zip(*ttt)]
                acc.append(mapped)
            all_successors=acc


            for beam_index, successors in enumerate(all_successors):
                new_beam = []
                for successor in successors:
                    if successor.last_action == 0 or t == self.episode_len - 1:
                        completed[beam_index].append(successor)
                    else:
                        new_beam.append(successor)
                if len(completed[beam_index]) >= beam_size:
                    new_beam = []
                new_beams.append(new_beam)

            beams = new_beams

            world_states = [
                [inf_state.world_state for inf_state in beam]
                for beam in beams
            ]

            obs = [
                [inf_state.observation for inf_state in beam]
                for beam in beams
            ]

            # Early exit if all ended
            if not any(beam for beam in beams):
                break

        trajs = []

        for this_completed in completed:
            assert this_completed
            this_trajs = []
            for inf_state in sorted(this_completed, key=lambda t: t.score, reverse=True)[:beam_size]:
                path_states, path_observations, path_actions, path_scores, path_attentions = backchain_inference_states(inf_state)
                # this will have messed-up headings for (at least some) starting locations because of
                # discretization, so read from the observations instead
                ## path = [(obs.viewpointId, state.heading, state.elevation)
                ##         for state in path_states]
                trajectory = [path_element_from_observation(ob) for ob in path_observations]
                this_trajs.append({
                    'instr_id': path_observations[0]['instr_id'],
                    'instr_encoding': path_observations[0]['instr_encoding'],
                    'path': trajectory,
                    'observations': path_observations,
                    'actions': path_actions,
                    'score': inf_state.score,
                    'scores': path_scores,
                    'attentions': path_attentions
                })
            trajs.append(this_trajs)
        traversed_lists = None  # todo
        return trajs, completed, traversed_lists


    def rollout_with_loss(self):
        world_states = self.env.reset(False)
        obs = np.array(self.env._get_obs(world_states))
        batch_size = len(obs)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]
        # when there are multiple sentences, perm_idx will simply be range(batch_size).
        # but perm_idx for each batch of i-th(i=1 or 2 or 3) will be inside seq_length.
        # this means:
        # seq_lengths=[(seq_lengths,perm_idx),(seq_lengths,perm_idx),(seq_lengths,perm_idx)]

        # Record starting point
        traj = [{'instr_id': ob['instr_id'], 'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]} for ob in perm_obs]

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx, h_t, c_t, seq_mask = self.encoder(seq, seq_mask, seq_lengths)

        if not self.decoder_init:
            #h_t, c_t = torch.zeros_like(h_t), torch.zeros_like(c_t)
            c_t = torch.zeros_like(c_t) # debug

        # Initial action
        a_t_prev, u_t_prev = None, None  # different action space
        if self.decoder.action_space == -1:
            u_t_prev = self.decoder.u_begin.expand(batch_size, -1)
        else:
            a_t_prev = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'), requires_grad=False).to(device)

        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Do a sequence rollout and calculate the loss
        self.loss = 0
        self.loss_ctrl_f = 0

        env_action = [None] * batch_size

        # for plot
        #all_alpha = []
        action_scores = np.zeros((batch_size,))

        for t in range(self.episode_len):
            f_t = self._feature_variable(perm_obs)

            # Image features from obs
            if self.decoder.panoramic: f_t_all, f_t = f_t
            else: f_t_all = np.zeros((batch_size, 1))

            # Action feature from obs
            if self.decoder.action_space == 6:
                u_t_features, is_valid = np.zeros((batch_size, 1)), None
            else:
                u_t_features, is_valid, _ = self._action_variable(perm_obs)

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            h_t, c_t, alpha, logit, pred_f = self.decoder(a_t_prev, u_t_prev, u_t_features, f_t, f_t_all, h_t, c_t, ctx, seq_mask)

            # all_alpha.append(alpha)
            # Mask outputs where agent can't move forward
            if self.decoder.action_space == 6:
                for i,ob in enumerate(perm_obs):
                    if len(ob['navigableLocations']) <= 1:
                        logit[i, self.model_actions.index('forward')] = -float('inf')
            else:
                logit[is_valid == 0] = -float('inf')
                # if self.decoder.ctrl_feature:
                #     pred_f[is_valid == 0] = 0

            if self.temp_alpha != 0: # add temperature
                logit = logit/self.temp_alpha

            self.loss += self.criterion(logit, target)

            # Auxiliary training
            if self.decoder.ctrl_feature:
                target_f = self._teacher_feature(perm_obs, ended)#, is_valid.shape[-1])
                self.loss_ctrl_f += self.aux_ratio * self.criterion_ctrl_f(pred_f, target_f)
            # todo: add auxiliary tasks to sc-rl training?

            # Determine next model inputs
            # scheduled sampling
            if self.schedule_ratio >= 0 and self.schedule_ratio <= 1:
                sample_feedback = random.choices(['sample', 'teacher'], [self.schedule_ratio, 1 - self.schedule_ratio], k=1)[0]  # schedule sampling
                if self.feedback != 'argmax': # ignore test case
                    self.feedback = sample_feedback

            a_t = self.get_next(self.feedback, target, logit)

            # setting A
            if self.testing_settingA:
                log_probs = F.log_softmax(logit, dim=1).data
                action_score = log_probs[torch.arange(batch_size), a_t].cpu().data.numpy()

            # log_probs = F.log_softmax(logit, dim=1).data
            # action_score = log_probs[torch.arange(batch_size), a_t].cpu().data.numpy()

            a_t_prev = a_t
            if self.decoder.action_space != 6:  # update the previous action
                u_t_prev = u_t_features[np.arange(batch_size), a_t, :].detach()

            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                if self.decoder.action_space == 6:
                    if action_idx == self.model_actions.index('<end>'):
                    #     # sub_stages[idx] = max(sub_stages[idx]-1, 0)
                    #     # ended[i] = (sub_stages[idx]==0)
                        ended[i] = True
                    env_action[idx] = self.env_actions[action_idx]
                else:
                    if action_idx == 0:
                    #     # sub_stages[idx] = max(sub_stages[idx] - 1, 0)
                    #     # ended[i] = (sub_stages[idx] == 0)
                        ended[i] = True
                    env_action[idx] = action_idx

            # state transitions
            new_states = self.env.step(env_action, obs)
            obs = np.array(self.env._get_obs(new_states))
            #obs = np.array(self.env._get_obs(self.env.step(env_action, obs)))  # , sub_stages
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()

                if self.testing_settingA:
                    if not ended[i]:
                        action_scores[idx] = action_scores[idx] + action_score[i]

                if self.decoder.action_space == 6:
                    if action_idx == self.model_actions.index('<end>'):
                        ended[i] = True
                else:
                    if action_idx == 0:
                        ended[i] = True

            # for i,idx in enumerate(perm_idx):
            #     action_idx = a_t[i].item()
            #     # if not ended[i]:
            #     #     action_scores[idx] = action_scores[idx] + action_score[i]
            #     if self.decoder.action_space == 6:
            #         if action_idx == self.model_actions.index('<end>'):
            #             ended[i] = True
            #     else:
            #         if action_idx == 0:
            #             ended[i] = True

            # Early exit if all ended
            if ended.all(): break

        # episode_len is just a constant so it doesn't matter
        self.losses.append(self.loss.item())  # / self.episode_len)
        if self.decoder.ctrl_feature:
            self.losses_ctrl_f.append(self.loss_ctrl_f.item())  # / self.episode_len)

        # with open('preprocess/alpha.pkl', 'wb') as alpha_f:  # TODO: remove for release!!!!
        #     pickle.dump(all_alpha, alpha_f)

        # chooes one from three according to prob
        # for t,p in zip(traj, action_scores):
        #     t['prob'] = p

        # chooes one from three according to prob
        if self.testing_settingA:
            for t, p in zip(traj, action_scores):
                t['prob'] = p

        return traj


    def img_shrink(self, feat_all):
        feat_dim = feat_all.shape[-1]
        f_t, act_t = feat_all[:,:, :feat_dim-128], feat_all[:,:,-128:]
        shrink = torch.cat([f_t, act_t, act_t], -1)[:,:,::3]
        return shrink


    def bert_rollout_with_loss(self):
        world_states = self.env.reset(False)
        obs = np.array(self.env._get_obs(world_states))
        batch_size = len(obs)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]
        # when there are multiple sentences, perm_idx will simply be range(batch_size).
        # but perm_idx for each batch of i-th(i=1 or 2 or 3) will be inside seq_length.
        # this means:
        # seq_lengths=[(seq_lengths,perm_idx),(seq_lengths,perm_idx),(seq_lengths,perm_idx)]

        # Record starting point
        traj = [{'instr_id': ob['instr_id'], 'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]} for ob in perm_obs]

        ## Forward through encoder, giving initial hidden state and memory cell for decoder
        #f_t = self._feature_variable(perm_obs)
        #if self.decoder.panoramic: f_t_all, f_t = f_t
        #else: f_t_all = np.zeros((batch_size, 1))
        #ctx, h_t, c_t, seq_mask = self.encoder(seq, seq_mask, seq_lengths,f_t_all=f_t_all)
        ###ctx, en_ht, en_ct, vl_mask = self.encoder(seq, seq_mask, seq_lengths,f_t_all=f_t_all)
        #if not self.decoder_init:
        #    h_t, c_t = torch.zeros_like(h_t), torch.zeros_like(c_t)
        #    #c_t = torch.zeros_like(c_t) # debug

        # Initial action
        a_t_prev, u_t_prev = None, None  # different action space
        if self.decoder.action_space == -1:
            u_t_prev = self.decoder.u_begin.expand(batch_size, -1)
        else:
            a_t_prev = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'), requires_grad=False).to(device)

        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Do a sequence rollout and calculate the loss
        self.loss = 0
        self.loss_ctrl_f = 0

        env_action = [None] * batch_size

        # for plot
        #all_alpha = []
        action_scores = np.zeros((batch_size,))
        for t in range(self.episode_len):
            f_t = self._feature_variable(perm_obs)

            # Image features from obs
            if self.decoder.panoramic: f_t_all, f_t = f_t
            else: f_t_all = np.zeros((batch_size, 1))

            # Action feature from obs
            if self.decoder.action_space == 6:
                u_t_features, is_valid = np.zeros((batch_size, 1)), None
            else:
                u_t_features, is_valid, _ = self._action_variable(perm_obs)

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            if self.encoder.__class__.__name__ in ['MultiVilAddEncoder','MultiAddLoadEncoder','MultiHugAddEncoder','MultiVicEncoder','MultiDicEncoder']:
                ctx, en_ht, en_ct, vl_mask = self.encoder(seq, seq_mask, seq_lengths, f_t_all=f_t_all)
            else:
                ctx, en_ht, en_ct, vl_mask = self.encoder(seq, seq_mask, torch.tensor(seq_lengths), f_t_all=f_t_all)


            if t == 0: # use encoder's ht and ct as init
                de_ht, de_ct, alpha, logit, pred_f = self.decoder(a_t_prev, u_t_prev, u_t_features, f_t, f_t_all, en_ht, en_ct, ctx, vl_mask)
            else:  # otherwise unroll as lstm
                de_ht, de_ct, alpha, logit, pred_f = self.decoder(a_t_prev, u_t_prev, u_t_features, f_t, f_t_all, de_ht, de_ct, ctx, vl_mask)


            # all_alpha.append(alpha)
            # Mask outputs where agent can't move forward
            if self.decoder.action_space == 6:
                for i,ob in enumerate(perm_obs):
                    if len(ob['navigableLocations']) <= 1:
                        logit[i, self.model_actions.index('forward')] = -float('inf')
            else:
                logit[is_valid == 0] = -float('inf')
                # if self.decoder.ctrl_feature:
                #     pred_f[is_valid == 0] = 0

            if self.temp_alpha != 0: # add temperature
                logit = logit/self.temp_alpha

            self.loss += self.criterion(logit, target)

            # Auxiliary training
            if self.decoder.ctrl_feature:
                target_f = self._teacher_feature(perm_obs, ended)#, is_valid.shape[-1])
                self.loss_ctrl_f += self.aux_ratio * self.criterion_ctrl_f(pred_f, target_f)
            # todo: add auxiliary tasks to sc-rl training?

            # Determine next model inputs
            # scheduled sampling
            if self.schedule_ratio >= 0 and self.schedule_ratio <= 1:
                sample_feedback = random.choices(['sample', 'teacher'], [self.schedule_ratio, 1 - self.schedule_ratio], k=1)[0]  # schedule sampling
                if self.feedback != 'argmax': # ignore test case
                    self.feedback = sample_feedback

            a_t = self.get_next(self.feedback, target, logit)

            # setting A
            if self.testing_settingA:
                log_probs = F.log_softmax(logit, dim=1).data
                action_score = log_probs[torch.arange(batch_size), a_t].cpu().data.numpy()

            # log_probs = F.log_softmax(logit, dim=1).data
            # action_score = log_probs[torch.arange(batch_size), a_t].cpu().data.numpy()

            a_t_prev = a_t
            if self.decoder.action_space != 6:  # update the previous action
                u_t_prev = u_t_features[np.arange(batch_size), a_t, :].detach()

            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                if self.decoder.action_space == 6:
                    if action_idx == self.model_actions.index('<end>'):
                    #     # sub_stages[idx] = max(sub_stages[idx]-1, 0)
                    #     # ended[i] = (sub_stages[idx]==0)
                        ended[i] = True
                    env_action[idx] = self.env_actions[action_idx]
                else:
                    if action_idx == 0:
                    #     # sub_stages[idx] = max(sub_stages[idx] - 1, 0)
                    #     # ended[i] = (sub_stages[idx] == 0)
                        ended[i] = True
                    env_action[idx] = action_idx

            # state transitions
            new_states = self.env.step(env_action, obs)
            obs = np.array(self.env._get_obs(new_states))
            #obs = np.array(self.env._get_obs(self.env.step(env_action, obs)))  # , sub_stages
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()

                if self.testing_settingA:
                    if not ended[i]:
                        action_scores[idx] = action_scores[idx] + action_score[i]

                if self.decoder.action_space == 6:
                    if action_idx == self.model_actions.index('<end>'):
                        ended[i] = True
                else:
                    if action_idx == 0:
                        ended[i] = True

            # for i,idx in enumerate(perm_idx):
            #     action_idx = a_t[i].item()
            #     # if not ended[i]:
            #     #     action_scores[idx] = action_scores[idx] + action_score[i]
            #     if self.decoder.action_space == 6:
            #         if action_idx == self.model_actions.index('<end>'):
            #             ended[i] = True
            #     else:
            #         if action_idx == 0:
            #             ended[i] = True

            # Early exit if all ended
            if ended.all(): break

        # episode_len is just a constant so it doesn't matter
        self.losses.append(self.loss.item())  # / self.episode_len)
        if self.decoder.ctrl_feature:
            self.losses_ctrl_f.append(self.loss_ctrl_f.item())  # / self.episode_len)

        # with open('preprocess/alpha.pkl', 'wb') as alpha_f:  # TODO: remove for release!!!!
        #     pickle.dump(all_alpha, alpha_f)

        # chooes one from three according to prob
        # for t,p in zip(traj, action_scores):
        #     t['prob'] = p

        # chooes one from three according to prob
        if self.testing_settingA:
            for t, p in zip(traj, action_scores):
                t['prob'] = p

        return traj

    def langbert_rollout_with_loss(self):
        world_states = self.env.reset(False)
        obs = np.array(self.env._get_obs(world_states))
        batch_size = len(obs)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]
        # when there are multiple sentences, perm_idx will simply be range(batch_size).
        # but perm_idx for each batch of i-th(i=1 or 2 or 3) will be inside seq_length.
        # this means:
        # seq_lengths=[(seq_lengths,perm_idx),(seq_lengths,perm_idx),(seq_lengths,perm_idx)]

        # Record starting point
        traj = [{'instr_id': ob['instr_id'], 'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]} for ob in perm_obs]

        ## Forward through encoder, giving initial hidden state and memory cell for decoder
        #f_t = self._feature_variable(perm_obs)
        #if self.decoder.panoramic: f_t_all, f_t = f_t
        #else: f_t_all = np.zeros((batch_size, 1))
        #ctx, h_t, c_t, seq_mask = self.encoder(seq, seq_mask, seq_lengths,f_t_all=f_t_all)
        ctx, en_ht, en_ct, vl_mask = self.encoder(seq, seq_mask, seq_lengths,f_t_all=None)
        #if not self.decoder_init:
        #    h_t, c_t = torch.zeros_like(h_t), torch.zeros_like(c_t)
        #    #c_t = torch.zeros_like(c_t) # debug

        # Initial action
        a_t_prev, u_t_prev = None, None  # different action space
        if self.decoder.action_space == -1:
            u_t_prev = self.decoder.u_begin.expand(batch_size, -1)
        else:
            a_t_prev = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'), requires_grad=False).to(device)

        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Do a sequence rollout and calculate the loss
        self.loss = 0
        self.loss_ctrl_f = 0

        env_action = [None] * batch_size

        # for plot
        #all_alpha = []
        action_scores = np.zeros((batch_size,))
        for t in range(self.episode_len):
            f_t = self._feature_variable(perm_obs)

            # Image features from obs
            if self.decoder.panoramic: f_t_all, f_t = f_t
            else: f_t_all = np.zeros((batch_size, 1))

            # Action feature from obs
            if self.decoder.action_space == 6:
                u_t_features, is_valid = np.zeros((batch_size, 1)), None
            else:
                u_t_features, is_valid, _ = self._action_variable(perm_obs)

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            #ctx, en_ht, en_ct, vl_mask = self.encoder(seq, seq_mask, torch.tensor(seq_lengths), f_t_all=f_t_all)

            if t == 0: # use encoder's ht and ct as init
                de_ht, de_ct, alpha, logit, pred_f = self.decoder(a_t_prev, u_t_prev, u_t_features, f_t, f_t_all, en_ht, en_ct, ctx, vl_mask)
            else:  # otherwise unroll as lstm
                de_ht, de_ct, alpha, logit, pred_f = self.decoder(a_t_prev, u_t_prev, u_t_features, f_t, f_t_all, de_ht, de_ct, ctx, vl_mask)


            # all_alpha.append(alpha)
            # Mask outputs where agent can't move forward
            if self.decoder.action_space == 6:
                for i,ob in enumerate(perm_obs):
                    if len(ob['navigableLocations']) <= 1:
                        logit[i, self.model_actions.index('forward')] = -float('inf')
            else:
                logit[is_valid == 0] = -float('inf')
                # if self.decoder.ctrl_feature:
                #     pred_f[is_valid == 0] = 0

            if self.temp_alpha != 0: # add temperature
                logit = logit/self.temp_alpha

            self.loss += self.criterion(logit, target)

            # Auxiliary training
            if self.decoder.ctrl_feature:
                target_f = self._teacher_feature(perm_obs, ended)#, is_valid.shape[-1])
                self.loss_ctrl_f += self.aux_ratio * self.criterion_ctrl_f(pred_f, target_f)
            # todo: add auxiliary tasks to sc-rl training?

            # Determine next model inputs
            # scheduled sampling
            if self.schedule_ratio >= 0 and self.schedule_ratio <= 1:
                sample_feedback = random.choices(['sample', 'teacher'], [self.schedule_ratio, 1 - self.schedule_ratio], k=1)[0]  # schedule sampling
                if self.feedback != 'argmax': # ignore test case
                    self.feedback = sample_feedback

            a_t = self.get_next(self.feedback, target, logit)

            # setting A
            if self.testing_settingA:
                log_probs = F.log_softmax(logit, dim=1).data
                action_score = log_probs[torch.arange(batch_size), a_t].cpu().data.numpy()

            # log_probs = F.log_softmax(logit, dim=1).data
            # action_score = log_probs[torch.arange(batch_size), a_t].cpu().data.numpy()

            a_t_prev = a_t
            if self.decoder.action_space != 6:  # update the previous action
                u_t_prev = u_t_features[np.arange(batch_size), a_t, :].detach()

            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                if self.decoder.action_space == 6:
                    if action_idx == self.model_actions.index('<end>'):
                    #     # sub_stages[idx] = max(sub_stages[idx]-1, 0)
                    #     # ended[i] = (sub_stages[idx]==0)
                        ended[i] = True
                    env_action[idx] = self.env_actions[action_idx]
                else:
                    if action_idx == 0:
                    #     # sub_stages[idx] = max(sub_stages[idx] - 1, 0)
                    #     # ended[i] = (sub_stages[idx] == 0)
                        ended[i] = True
                    env_action[idx] = action_idx

            # state transitions
            new_states = self.env.step(env_action, obs)
            obs = np.array(self.env._get_obs(new_states))
            #obs = np.array(self.env._get_obs(self.env.step(env_action, obs)))  # , sub_stages
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()

                if self.testing_settingA:
                    if not ended[i]:
                        action_scores[idx] = action_scores[idx] + action_score[i]

                if self.decoder.action_space == 6:
                    if action_idx == self.model_actions.index('<end>'):
                        ended[i] = True
                else:
                    if action_idx == 0:
                        ended[i] = True

            # for i,idx in enumerate(perm_idx):
            #     action_idx = a_t[i].item()
            #     # if not ended[i]:
            #     #     action_scores[idx] = action_scores[idx] + action_score[i]
            #     if self.decoder.action_space == 6:
            #         if action_idx == self.model_actions.index('<end>'):
            #             ended[i] = True
            #     else:
            #         if action_idx == 0:
            #             ended[i] = True

            # Early exit if all ended
            if ended.all(): break

        # episode_len is just a constant so it doesn't matter
        self.losses.append(self.loss.item())  # / self.episode_len)
        if self.decoder.ctrl_feature:
            self.losses_ctrl_f.append(self.loss_ctrl_f.item())  # / self.episode_len)

        # with open('preprocess/alpha.pkl', 'wb') as alpha_f:  # TODO: remove for release!!!!
        #     pickle.dump(all_alpha, alpha_f)

        # chooes one from three according to prob
        # for t,p in zip(traj, action_scores):
        #     t['prob'] = p

        # chooes one from three according to prob
        if self.testing_settingA:
            for t, p in zip(traj, action_scores):
                t['prob'] = p

        return traj


    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, beam_size=1, successors=1, speaker=(None,None,None,None)):
        ''' Evaluate once on each instruction in the current environment '''
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        super(Seq2SeqAgent, self).test(beam_size, successors, speaker)

    def train(self, encoder_optimizer, decoder_optimizer, n_iters, aux_n_iter, feedback='teacher'):
        ''' Train for a given number of iterations '''
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        self.losses_ctrl_f = []
        epo_inc = 0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        for iter in range(1, n_iters + 1):
            if self.accu_n_iters == 0:
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                self.rollout()
                if (not self.decoder.ctrl_feature) or (iter % aux_n_iter):
                    self.loss.backward()
                else:
                    self.loss_ctrl_f.backward()

                if self.clip_gradient != 0: # clip gradient
                    clip_gradient(encoder_optimizer, self.clip_gradient)
                    clip_gradient(decoder_optimizer, self.clip_gradient)

                if self.clip_gradient_norm > 0: # clip gradient norm
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_gradient_norm)
                    torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip_gradient_norm)

                encoder_optimizer.step()
                decoder_optimizer.step()

                epo_inc += self.env.epo_inc

            else:
                self.rollout()
                if (not self.decoder.ctrl_feature) or (iter % aux_n_iter):
                    self.loss /= self.accu_n_iters
                    self.loss.backward()
                else:
                    self.loss_ctrl_f.backward()

                if iter % self.accu_n_iters == 0:
                    if self.clip_gradient != 0: # clip gradient
                        clip_gradient(encoder_optimizer, self.clip_gradient)
                        clip_gradient(decoder_optimizer, self.clip_gradient)

                    if self.clip_gradient_norm > 0: # clip gradient norm
                        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_gradient_norm)
                        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip_gradient_norm)

                    encoder_optimizer.step()
                    decoder_optimizer.step()

                    encoder_optimizer.zero_grad()
                    decoder_optimizer.zero_grad()
                epo_inc += self.env.epo_inc

        return epo_inc

    """
    def train(self, encoder_optimizer, decoder_optimizer, n_iters, aux_n_iter, feedback='teacher'):
        ''' Train for a given number of iterations '''
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        self.losses_ctrl_f = []
        epo_inc = 0
        for iter in range(1, n_iters + 1):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            self.rollout()
            if (not self.decoder.ctrl_feature) or (iter % aux_n_iter):
                self.loss.backward()
            else:
                self.loss_ctrl_f.backward()

            if self.clip_gradient != 0: # clip gradient
                clip_gradient(encoder_optimizer, self.clip_gradient)
                clip_gradient(decoder_optimizer, self.clip_gradient)

            if self.clip_gradient_norm > 0: # clip gradient norm
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_gradient_norm)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip_gradient_norm)

            encoder_optimizer.step()
            decoder_optimizer.step()

            epo_inc += self.env.epo_inc
        return epo_inc
    """

    def rollout_notrain(self, n_iters):  # jolin
        epo_inc = 0
        for iter in range(1, n_iters + 1):
            self.env._next_minibatch(False)
            epo_inc += self.env.epo_inc
        return epo_inc

    def rl_rollout(self, obs, perm_obs, seq, seq_mask, seq_lengths, perm_idx, feedback,
                   encoder, decoder):
        batch_size = len(perm_obs)
        # Record starting point
        traj = [{'instr_id': ob['instr_id'],
                 'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx, h_t, c_t, seq_mask = encoder(seq, seq_mask, seq_lengths)
        if not self.decoder_init:
            h_t, c_t = torch.zeros_like(h_t), torch.zeros_like(c_t)

        # Initial action
        a_t_prev, u_t_prev = None, None  # different action space
        if decoder.action_space==-1:
            u_t_prev = decoder.u_begin.expand(batch_size, -1)
        else:
            a_t_prev = Variable(torch.ones(batch_size).long() *
                self.model_actions.index('<start>'), requires_grad=False).to(device)

        ended=np.array([False] * batch_size)

        # Do a sequence rollout not don't calculate the loss for policy gradient
        # self.loss = 0
        env_action = [None] * batch_size

        # Initialize seq log probs for policy gradient
        if feedback == 'sample1':
            seqLogprobs = h_t.new_zeros(batch_size, self.episode_len)
            mask = np.ones((batch_size, self.episode_len))
        elif feedback == 'argmax1':
            seqLogprobs, mask = None, None
        else:
            raise NotImplementedError('other feedback not supported.')

        # only for supervised auxiliary tasks
        #assert (not self.decoder.ctrl_feature)  # not implemented
        for t in range(self.episode_len):
            f_t = self._feature_variable(perm_obs)

            # Image features from obs
            if decoder.panoramic: f_t_all, f_t = f_t
            else: f_t_all = np.zeros((batch_size, 1))

            # Action feature from obs
            if decoder.action_space == 6:
                u_t_features, is_valid = np.zeros((batch_size, 1)), None
            else:
                u_t_features, is_valid, _ = self._action_variable(perm_obs)

            # Decoding actions
            h_t, c_t, alpha, logit, pred_f = decoder(a_t_prev, u_t_prev, u_t_features,
                                               f_t, f_t_all, h_t, c_t, ctx, seq_mask)

            # Mask outputs where agent can't move forward
            if decoder.action_space == 6:
                for i,ob in enumerate(perm_obs):
                    if len(ob['navigableLocations']) <= 1:
                        logit[i, self.model_actions.index('forward')] = -float('inf')
            else: logit[is_valid == 0] = -float('inf')

            # Supervised training
            # target = self._teacher_action(perm_obs, ended)
            # self.loss += self.criterion(logit, target)

            # Determine next model inputs
            if feedback == 'argmax1':
                _, a_t = logit.max(1)
            elif feedback == 'sample1':
                logprobs = F.log_softmax(logit, dim=1)
                probs = torch.exp(logprobs.data)
                m = D.Categorical(probs)
                a_t = m.sample()  # sampling an action from model
                sampleLogprobs = logprobs.gather(1, a_t.unsqueeze(1))
            else:
                sys.exit('invalid feedback method %s'%feedback)
            # if self.decoder.panoramic:
            #     a_t_feature = all_a_t_features[np.arange(batch_size), a_t, :].detach()

            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                if decoder.action_space == 6:
                    if ended[i] and mask is not None:
                            mask[i, t] = 0
                    elif action_idx == self.model_actions.index('<end>'):
                        ended[i] = True
                    env_action[idx] = self.env_actions[action_idx]
                else:
                    if ended[i] and mask is not None:
                            mask[i, t] = 0
                    elif action_idx == 0: ended[i] = True
                    env_action[idx] = action_idx

            obs = np.array(self.env._get_obs(self.env.step(env_action, obs)))
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'],
                                            ob['heading'], ob['elevation']))

            if seqLogprobs is not None:  # feedback == 'sample1'
                seqLogprobs[:, t] = sampleLogprobs.view(-1)

            # Early exit if all ended
            if ended.all(): break

        path_res = {}
        for t in traj:
            path_res[t['instr_id']] = t['path']
        return traj, mask, seqLogprobs, path_res

    def rl_train(self, train_Eval, encoder_optimizer, decoder_optimizer,
                 n_iters, sc_reward_scale, sc_discouted_immediate_r_scale, sc_length_scale):
        ''' jolin: self-critical finetuning'''
        self.losses = []
        epo_inc = 0
        self.encoder.train()
        self.decoder.train()
        for iter in range(1, n_iters + 1):  # n_iters=interval
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            # copy from self.rollout():
            # one minibatch (100 instructions)
            world_states = self.env.reset(False)
            obs = np.array(self.env._get_obs(world_states))
            epo_inc += self.env.epo_inc
            # Reorder the language input for the encoder
            seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
            perm_obs = obs[perm_idx]
            gen_traj, mask, seqLogprobs, gen_results = self.rl_rollout(obs, perm_obs, seq, seq_mask,
                                                             seq_lengths, perm_idx, 'sample1',
                                                             self.encoder, self.decoder)

            # jolin: get greedy decoding baseline
            # Just like self.test(use_dropout=False, feedback='argmax').
            # But we should not do env.reset_epoch(), because we do not
            # test the whole split. So DON'T reuse test()!

            world_states = self.env.reset_batch()
            obs = np.array(self.env._get_obs(world_states))# for later 'sample' feedback batch
            perm_obs = obs[perm_idx]

            if self.monotonic:
                encoder2, decoder2 = self.encoder2, self.decoder2
            else:
                self.encoder.eval()
                self.decoder.eval()
                encoder2, decoder2 = self.encoder, self.decoder
            with torch.no_grad():
                greedy_traj, _, _, greedy_res = self.rl_rollout(obs, perm_obs, seq, seq_mask,
                                                      seq_lengths, perm_idx, 'argmax1',
                                                      encoder2, decoder2)
            if not self.monotonic:
                self.encoder.train()
                self.decoder.train()

            # jolin: get self-critical reward
            reward = self.get_self_critical_reward(gen_traj, train_Eval, gen_results, greedy_res, sc_reward_scale, sc_discouted_immediate_r_scale, sc_length_scale)

            # jolin: RewardCriterion
            self.loss = self.PG_reward_criterion(seqLogprobs, reward, mask)

            self.losses.append(self.loss.item())
            self.loss.backward()
            #clip_gradient(encoder_optimizer)
            #clip_gradient(decoder_optimizer)

            if self.clip_gradient != 0: # clip gradient
                clip_gradient(encoder_optimizer, self.clip_gradient)
                clip_gradient(decoder_optimizer, self.clip_gradient)

            if self.clip_gradient_norm > 0: # clip gradient norm
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_gradient_norm)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip_gradient_norm)

            encoder_optimizer.step()
            decoder_optimizer.step()
        return epo_inc

    def PG_reward_criterion(self, seqLogprobs, reward, mask):
        # jolin: RewardCriterion
        input = to_contiguous(seqLogprobs).view(-1)
        reward = to_contiguous(torch.from_numpy(reward).float().to(device)).view(-1)
        #mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        mask = to_contiguous(torch.from_numpy(mask).float().to(device)).view(-1)
        output = - input * reward * mask
        loss = torch.sum(output) / torch.sum(mask)
        return loss

    def get_self_critical_reward(self, traj, Eval, gen_results, greedy_res, sc_reward_scale, sc_discouted_immediate_r_scale, sc_length_scale):
        # get self-critical reward
        instr_id_order = [t['instr_id'] for t in traj]
        gen_scores = Eval.score_batch(gen_results, instr_id_order)
        greedy_scores = Eval.score_batch(greedy_res, instr_id_order)

        # normal score
        gen_hits = (np.array(gen_scores['nav_errors']) <= 3.0).astype(float)
        greedy_hits = (np.array(greedy_scores['nav_errors']) <= 3.0).astype(float)
        gen_lengths = (np.array(gen_scores['trajectory_lengths'])).astype(float)

        # sr_sc
        hits = gen_hits - greedy_hits
        #reward = np.repeat(hits[:, np.newaxis], self.episode_len, 1)*sc_reward_scale

        # spl
        gen_spls = (np.array(gen_scores['spl'])).astype(float)
        greedy_spls = (np.array(greedy_scores['spl'])).astype(float)

        ave_steps = (np.array(gen_scores['trajectory_steps'])).sum()/float(len(instr_id_order))
        steps = (np.array(gen_scores['trajectory_steps']) - self.episode_len).sum()

        if self.reward_func == 'sr_sc':
            reward = np.repeat(hits[:, np.newaxis], self.episode_len, 1)*sc_reward_scale
        elif self.reward_func == 'spl':
            reward = np.repeat(gen_spls[:, np.newaxis], self.episode_len, 1) * sc_reward_scale
        elif self.reward_func == 'spl_sc':
            # spl_sc
            spl_sc = gen_spls - greedy_spls
            reward = np.repeat(spl_sc[:, np.newaxis], self.episode_len, 1) * sc_reward_scale
        elif self.reward_func == 'spl_last': # does not work
            tj_steps = [s - 1 for s in gen_scores['trajectory_steps']] # tj_steps
            reward = np.zeros((gen_spls.shape[0], self.episode_len), dtype=float)
            reward[range(gen_spls.shape[0]), tj_steps] = gen_scores['spl']
        elif self.reward_func == 'spl_last_sc':
            tj_steps = [s - 1 for s in gen_scores['trajectory_steps']] # tj_steps
            reward = np.zeros((gen_spls.shape[0], self.episode_len), dtype=float)
            reward[range(gen_spls.shape[0]), tj_steps] = [x - y for x, y in zip(gen_scores['spl'], greedy_scores['spl'])]
        elif self.reward_func == 'spl_psc': # test
            tj_steps = [s - 1 for s in gen_scores['trajectory_steps']]  # tj_steps
            reward = np.full((gen_spls.shape[0], self.episode_len), -sc_length_scale, dtype=float) # penalty
            reward[range(gen_spls.shape[0]), tj_steps] = gen_scores['spl']

        # discounted immediate reward
        if sc_discouted_immediate_r_scale>0:
            discounted_r = discount_rewards(gen_scores['immediate_rewards'], self.episode_len) * sc_discouted_immediate_r_scale
            reward = reward + discounted_r

        # panelty for length
        if sc_length_scale:
            length_panelty = np.repeat(gen_lengths[:, np.newaxis], self.episode_len, 1)*sc_length_scale
            reward = reward - length_panelty
        return reward


    def save(self, encoder_path, decoder_path):
        ''' Snapshot models '''

        write_num = 0
        while (write_num < 10):
            try:
                torch.save(self.encoder.state_dict(), encoder_path)
                torch.save(self.decoder.state_dict(), decoder_path)
                if torch.cuda.is_available():
                    torch.save(torch.cuda.random.get_rng_state(), decoder_path + '.rng.gpu')
                torch.save(torch.random.get_rng_state(), decoder_path + '.rng')
                with open(decoder_path + '.rng2', 'wb') as f:
                    pickle.dump(random.getstate(), f)
                break
            except:
                write_num += 1

    def delete(self, encoder_path, decoder_path):
        ''' Delete models '''
        os.remove(encoder_path)
        os.remove(decoder_path)
        os.remove(decoder_path+'.rng.gpu')
        os.remove(decoder_path+'.rng')
        os.remove(decoder_path+'.rng2')

    def load(self, encoder_path, decoder_path):
        ''' Loads parameters (but not training state) '''
        self.encoder.load_state_dict(torch.load(encoder_path, 'cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.decoder.load_state_dict(torch.load(decoder_path, 'cuda:0' if torch.cuda.is_available() else 'cpu'), strict=False)
        self.encoder.to(device)
        self.decoder.to(device)
        if self.monotonic:
            self.copy_seq2seq()

        try:
            with open(decoder_path+'.rng2','rb') as f:
                random.setstate(pickle.load(f))
            torch.random.set_rng_state(torch.load(decoder_path + '.rng'))
            torch.cuda.random.set_rng_state(torch.load(decoder_path + '.rng.gpu'))
        except FileNotFoundError:
            print('Warning: failed to find random seed file')

    def copy_seq2seq(self):
        self.encoder2=copy.deepcopy(self.encoder)
        self.decoder2=copy.deepcopy(self.decoder)
        self.encoder2.eval()
        self.decoder2.eval()
        for param in self.encoder2.parameters():
            param.requires_grad = False
        for param in self.decoder2.parameters():
            param.requires_grad = False


class PretrainVLAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''
    model_actions, env_actions = basic_actions()
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, encoder, decoder, seed, aux_ratio, decoder_init,
                 params=None, monotonic=False, episode_len=20, state_factored=False):  # , subgoal
        super(Seq2SeqAgent, self).__init__(env, results_path, seed=seed)
        self.encoder, self.decoder = encoder, decoder  # encoder2 is only for self_critic
        self.encoder2, self.decoder2 = None, None
        self.monotonic = monotonic
        if self.monotonic:
            self.copy_seq2seq()
        self.episode_len = episode_len
        self.losses = []
        self.losses_ctrl_f = [] # For learning auxiliary tasks
        self.aux_ratio = aux_ratio
        self.decoder_init = decoder_init

        self.clip_gradient = params['clip_gradient']
        self.clip_gradient_norm = params['clip_gradient_norm']
        self.reward_func = params['reward_func']

        self.schedule_ratio = params['schedule_ratio']
        self.temp_alpha = params['temp_alpha']

        self.testing_settingA = params['test_A']

        if self.decoder.action_space == 6:
            self.ignore_index = self.model_actions.index('<ignore>')
        else:
            self.ignore_index = -1
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        if self.decoder.ctrl_feature:
            assert self.decoder.action_space == -1 # currently only implement this
            self.criterion_ctrl_f = nn.MSELoss()  # todo: MSE or ?

        self.state_factored = state_factored

    @staticmethod
    def n_inputs():
        return len(Seq2SeqAgent.model_actions)

    @staticmethod
    def n_outputs():
        return len(Seq2SeqAgent.model_actions)-2 # Model doesn't output start or ignore

    def _sort_batch(self, obs):
        sorted_tensor, mask, seq_lengths, perm_idx = sort_batch(obs)

        if isinstance(sorted_tensor, list):
            sorted_tensors, masks, seqs_lengths = [], [], []
            for i in range(len(sorted_tensor)):
                sorted_tensors.append(Variable(sorted_tensor[i], requires_grad=False).long().to(device))
                masks.append(mask[i].byte().to(device))
                seqs_lengths.append(seq_lengths[i])
            return sorted_tensors, masks, seqs_lengths, perm_idx

        return Variable(sorted_tensor, requires_grad=False).long().to(device), \
               mask.byte().to(device), \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        #feature_size = obs[0]['feature'].shape[0]
        #features = np.empty((len(obs),feature_size), dtype=np.float32)
        if isinstance(obs[0]['feature'],tuple): # todo?
            features_pano = np.repeat(np.expand_dims(np.zeros_like(obs[0]['feature'][0], dtype=np.float32), 0), len(obs), axis=0)  # jolin
            features = np.repeat(np.expand_dims(np.zeros_like(obs[0]['feature'][1], dtype=np.float32), 0), len(obs), axis=0)  # jolin
            for i,ob in enumerate(obs):
                features_pano[i] = ob['feature'][0]
                features[i] = ob['feature'][1]
            return (Variable(torch.from_numpy(features_pano), requires_grad=False).to(device),
            Variable(torch.from_numpy(features), requires_grad=False).to(device))
        else:
            features = np.repeat(np.expand_dims(np.zeros_like(obs[0]['feature'], dtype=np.float32),0),len(obs),axis=0)  # jolin
            for i,ob in enumerate(obs):
                features[i] = ob['feature']
            return Variable(torch.from_numpy(features), requires_grad=False).to(device)


    def get_next(self, feedback, target, logit):
        if feedback == 'teacher':
            a_t = target  # teacher forcing
        elif feedback == 'argmax':
            _, a_t = logit.max(1)  # student forcing - argmax
            a_t = a_t.detach()
        elif feedback == 'sample':
            probs = F.softmax(logit, dim=1)
            m = D.Categorical(probs)
            a_t = m.sample()  # sampling an action from model
        else:
            sys.exit('Invalid feedback option')
        return a_t

    def _action_variable(self, obs):
        # get the maximum number of actions of all sample in this batch
        max_num_a = -1
        for i, ob in enumerate(obs):
            max_num_a = max(max_num_a, len(ob['adj_loc_list']))

        is_valid = np.zeros((len(obs), max_num_a), np.float32)
        action_embedding_dim = obs[0]['action_embedding'].shape[-1]
        action_embeddings = np.zeros((len(obs), max_num_a, action_embedding_dim), dtype=np.float32)
        for i, ob in enumerate(obs):
            adj_loc_list = ob['adj_loc_list']
            num_a = len(adj_loc_list)
            is_valid[i, 0:num_a] = 1.
            action_embeddings[i, :num_a, :] = ob['action_embedding'] #bug: todo
            #for n_a, adj_dict in enumerate(adj_loc_list):
            #    action_embeddings[i, :num_a, :] = ob['action_embedding']
        return (Variable(torch.from_numpy(action_embeddings), requires_grad=False).to(device),
                Variable(torch.from_numpy(is_valid), requires_grad=False).to(device),
                is_valid)

    def _teacher_action_variable(self, obs):
        # get the maximum number of actions of all sample in this batch
        action_embedding_dim = obs[0]['action_embedding'].shape[-1]
        action_embeddings = np.zeros((len(obs), action_embedding_dim), dtype=np.float32)
        for i, ob in enumerate(obs):
            adj_loc_list = ob['adj_loc_list']
            action_embeddings[i, :] = ob['action_embedding'][ob['teacher']] #bug: todo
            #for n_a, adj_dict in enumerate(adj_loc_list):
            #    action_embeddings[i, :num_a, :] = ob['action_embedding']
        return Variable(torch.from_numpy(action_embeddings), requires_grad=False).to(device)

    def _teacher_action(self, obs, ended):
        a = teacher_action(self.model_actions, self.decoder.action_space, obs, ended, self.ignore_index)
        return Variable(a, requires_grad=False).to(device)

    def _teacher_feature(self, obs, ended):#, max_num_a):
        ''' Extract teacher look ahead auxiliary features into variable. '''
        # todo: 6 action space
        ctrl_features_dim = -1
        for i, ob in enumerate(obs):  # todo: whether include <stop> ?
            # max_num_a = max(max_num_a, len(ob['ctrl_features']))
            if ctrl_features_dim<0 and len(ob['ctrl_features']):
                ctrl_features_dim = ob['ctrl_features'].shape[-1] #[0].shape[-1]
                break

        #is_valid no need to create. already created
        ctrl_features_tensor = np.zeros((len(obs), ctrl_features_dim), dtype=np.float32)
        for i, ob in enumerate(obs):
            if not ended[i]:
                ctrl_features_tensor[i, :] = ob['ctrl_features']
        return Variable(torch.from_numpy(ctrl_features_tensor), requires_grad=False).to(device)

    def rollout(self, beam_size=1, successors=1):
        if beam_size ==1 and not debug_beam:
            if self.encoder.__class__.__name__ == 'BertImgEncoder':
                return self.pretrain_rollout_with_loss()
            else:
                return self.rollout_with_loss()

        # beam
        with torch.no_grad():
            if self.state_factored:
                beams = self.state_factored_search(beam_size, successors, first_n_ws_key=4)
            else:
                beams = self.beam_search(beam_size)
        return beams

    def state_factored_search(self, completion_size, successor_size, first_n_ws_key=4):
        assert self.decoder.panoramic
        world_states = self.env.reset(sort=True)
        initial_obs = (self.env._get_obs(world_states))
        batch_size = len(world_states)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch([o for ob in initial_obs for o in ob])

        world_states = [[world_state for f, world_state in states] for states in world_states]

        ctx, h_t, c_t, seq_mask = self.encoder(seq, seq_mask, seq_lengths)
        if not self.decoder_init:
            h_t, c_t = torch.zeros_like(h_t), torch.zeros_like(c_t)

        completed = []
        completed_holding = []
        for _ in range(batch_size):
            completed.append({})
            completed_holding.append({})

        state_cache = [
            {ws[0][0:first_n_ws_key]: (InferenceState(prev_inference_state=None,
                                                      world_state=ws[0],
                                                      observation=o[0],
                                                      flat_index=None,
                                                      last_action=-1,
                                                      last_action_embedding=self.decoder.u_begin,
                                                      action_count=0,
                                                      score=0.0, h_t=h_t[i], c_t=c_t[i], last_alpha=None), True)}
            for i, (ws, o) in enumerate(zip(world_states, initial_obs))
        ]

        beams = [[inf_state for world_state, (inf_state, expanded) in sorted(instance_cache.items())]
                 for instance_cache in state_cache] # sorting is a noop here since each instance_cache should only contain one

        last_expanded_list = []
        traversed_lists = []

        for beam in beams:
            assert len(beam)==1
            first_state = beam[0]
            last_expanded_list.append(first_state)
            traversed_lists.append([first_state])

        def update_traversed_lists(new_visited_inf_states):
            assert len(new_visited_inf_states) == len(last_expanded_list)
            assert len(new_visited_inf_states) == len(traversed_lists)

            for instance_index, instance_states in enumerate(new_visited_inf_states):
                last_expanded = last_expanded_list[instance_index]
                # todo: if this passes, shouldn't need traversed_lists
                assert last_expanded.world_state.viewpointId == traversed_lists[instance_index][-1].world_state.viewpointId
                for inf_state in instance_states:
                    path_from_last_to_next = least_common_viewpoint_path(last_expanded, inf_state)
                    # path_from_last should include last_expanded's world state as the first element, so check and drop that
                    assert path_from_last_to_next[0].world_state.viewpointId == last_expanded.world_state.viewpointId
                    assert path_from_last_to_next[-1].world_state.viewpointId == inf_state.world_state.viewpointId
                    traversed_lists[instance_index].extend(path_from_last_to_next[1:])
                    last_expanded = inf_state
                last_expanded_list[instance_index] = last_expanded


        # Do a sequence rollout and calculate the loss
        while any(len(comp) < completion_size for comp in completed):
            beam_indices = []
            u_t_list = []
            h_t_list = []
            c_t_list = []
            flat_obs = []
            for beam_index, beam in enumerate(beams):
                for inf_state in beam:
                    beam_indices.append(beam_index)
                    u_t_list.append(inf_state.last_action_embedding)
                    h_t_list.append(inf_state.h_t.unsqueeze(0))
                    c_t_list.append(inf_state.c_t.unsqueeze(0))
                    flat_obs.append(inf_state.observation)

            u_t_prev = torch.stack(u_t_list, dim=0)
            assert len(u_t_prev.shape) == 2
            # Image features from obs
            # if self.decoder.panoramic:
            f_t_all, f_t = self._feature_variable(flat_obs)


            # Action feature from obs
            # if self.decoder.action_space == 6:
            #     u_t_features, is_valid = np.zeros((batch_size, 1)), None
            # else:
            u_t_features, is_valid, is_valid_numpy = self._action_variable(flat_obs)

            h_t = torch.cat(h_t_list, dim=0)
            c_t = torch.cat(c_t_list, dim=0)

            h_t, c_t, alpha, logit, pred_f = self.decoder(None, u_t_prev, u_t_features, f_t,
                                                          f_t_all, h_t, c_t, [ctx_si[beam_indices] for ctx_si in ctx] if isinstance(ctx, list) else ctx[beam_indices],
                  [seq_mask_si[beam_indices] for seq_mask_si in seq_mask] if isinstance(ctx, list) else seq_mask[beam_indices])

            # Mask outputs of invalid actions
            logit[is_valid == 0] = -float('inf')
            # # Mask outputs where agent can't move forward
            # no_forward_mask = [len(ob['navigableLocations']) <= 1 for ob in flat_obs]

            masked_logit = logit
            log_probs = F.log_softmax(logit, dim=1).data

            # force ending if we've reached the max time steps
            # if t == self.episode_len - 1:
            #     action_scores = log_probs[:,self.end_index].unsqueeze(-1)
            #     action_indices = torch.from_numpy(np.full((log_probs.size()[0], 1), self.end_index))
            # else:
            #_, action_indices = masked_logit.data.topk(min(successor_size, logit.size()[1]), dim=1)
            _, action_indices = masked_logit.data.topk(logit.size()[1], dim=1) # todo: fix this
            action_scores = log_probs.gather(1, action_indices)
            assert action_scores.size() == action_indices.size()

            start_index = 0
            assert len(beams) == len(world_states)
            all_successors = []
            for beam_index, (beam, beam_world_states) in enumerate(zip(beams, world_states)):
                successors = []
                end_index = start_index + len(beam)
                assert len(beam_world_states) == len(beam)
                if beam:
                    for inf_index, (inf_state, world_state, action_score_row) in \
                            enumerate(zip(beam, beam_world_states, log_probs[start_index:end_index])):
                        flat_index = start_index + inf_index
                        for action_index, action_score in enumerate(action_score_row):
                            if is_valid_numpy[flat_index, action_index] == 0:
                                continue
                            successors.append(
                                InferenceState(prev_inference_state=inf_state, world_state=world_state,
                                               observation=flat_obs[flat_index],
                                               flat_index=None,
                                               last_action=action_index,
                                               last_action_embedding=u_t_features[flat_index, action_index].detach(),
                                               action_count=inf_state.action_count + 1,
                                               score=float(inf_state.score + action_score),
                                               h_t=h_t[flat_index], c_t=c_t[flat_index],
                                               last_alpha=[alpha_si[flat_index].data for alpha_si in alpha] if isinstance(alpha, list) else alpha[flat_index].data)
                            )
                start_index = end_index
                successors = sorted(successors, key=lambda t: t.score, reverse=True)
                all_successors.append(successors)

            successor_world_states = [
                [inf_state.world_state for inf_state in successors]
                for successors in all_successors
            ]

            successor_env_actions = [
                [inf_state.last_action for inf_state in successors]
                for successors in all_successors
            ]

            successor_last_obs = [
                [inf_state.observation for inf_state in successors]
                for successors in all_successors
            ]

            successor_world_states = self.env.step(successor_env_actions, successor_last_obs, successor_world_states)
            successor_world_states = [[world_state for f, world_state in states] for states in successor_world_states]

            acc = []
            for ttt in zip(all_successors, successor_world_states):
                mapped = [inf._replace(world_state=ws) for inf, ws in zip(*ttt)]
                acc.append(mapped)
            all_successors = acc

            assert len(all_successors) == len(state_cache)

            new_beams = []

            for beam_index, (successors, instance_cache) in enumerate(zip(all_successors, state_cache)):
                # early stop if we've already built a sizable completion list
                instance_completed = completed[beam_index]
                instance_completed_holding = completed_holding[beam_index]
                if len(instance_completed) >= completion_size:
                    new_beams.append([])
                    continue
                for successor in successors:
                    ws_keys = successor.world_state[0:first_n_ws_key]
                    if successor.last_action == 0 or successor.action_count == self.episode_len:
                        if ws_keys not in instance_completed_holding or instance_completed_holding[ws_keys][
                            0].score < successor.score:
                            instance_completed_holding[ws_keys] = (successor, False)
                    else:
                        if ws_keys not in instance_cache or instance_cache[ws_keys][0].score < successor.score:
                            instance_cache[ws_keys] = (successor, False)

                # third value: did this come from completed_holding?
                uncompleted_to_consider = ((ws_keys, inf_state, False) for (ws_keys, (inf_state, expanded)) in
                                           instance_cache.items() if not expanded)
                completed_to_consider = ((ws_keys, inf_state, True) for (ws_keys, (inf_state, expanded)) in
                                         instance_completed_holding.items() if not expanded)
                import itertools
                import heapq
                to_consider = itertools.chain(uncompleted_to_consider, completed_to_consider)
                ws_keys_and_inf_states = heapq.nlargest(successor_size, to_consider, key=lambda pair: pair[1].score)

                new_beam = []
                for ws_keys, inf_state, is_completed in ws_keys_and_inf_states:
                    if is_completed:
                        assert instance_completed_holding[ws_keys] == (inf_state, False)
                        instance_completed_holding[ws_keys] = (inf_state, True)
                        if ws_keys not in instance_completed or instance_completed[ws_keys].score < inf_state.score:
                            instance_completed[ws_keys] = inf_state
                    else:
                        instance_cache[ws_keys] = (inf_state, True)
                        new_beam.append(inf_state)

                if len(instance_completed) >= completion_size:
                    new_beams.append([])
                else:
                    new_beams.append(new_beam)

            beams = new_beams

            # Early exit if all ended
            if not any(beam for beam in beams):
                break

            world_states = [
                [inf_state.world_state for inf_state in beam]
                for beam in beams
            ]
            successor_obs = np.array(self.env._get_obs(self.env.world_states2feature_states(world_states)))

            acc = []
            for tttt in zip(beams, successor_obs):
                mapped = [inf._replace(observation=o) for inf, o in zip(*tttt)]
                acc.append(mapped)
            beams = acc
            update_traversed_lists(beams)

        completed_list = []
        for this_completed in completed:
            completed_list.append(sorted(this_completed.values(), key=lambda t: t.score, reverse=True)[:completion_size])
        completed_ws = [
            [inf_state.world_state for inf_state in comp_l]
            for comp_l in completed_list
        ]
        completed_obs = np.array(self.env._get_obs(self.env.world_states2feature_states(completed_ws)))
        accu = []
        for ttttt in zip(completed_list, completed_obs):
            mapped = [inf._replace(observation=o) for inf, o in zip(*ttttt)]
            accu.append(mapped)
        completed_list = accu

        update_traversed_lists(completed_list)

        trajs = []
        for this_completed in completed_list:
            assert this_completed
            this_trajs = []
            for inf_state in this_completed:
                path_states, path_observations, path_actions, path_scores, path_attentions = backchain_inference_states(inf_state)
                # this will have messed-up headings for (at least some) starting locations because of
                # discretization, so read from the observations instead
                ## path = [(obs.viewpointId, state.heading, state.elevation)
                ##         for state in path_states]
                trajectory = [path_element_from_observation(ob) for ob in path_observations]
                this_trajs.append({
                    'instr_id': path_observations[0]['instr_id'],
                    'instr_encoding': path_observations[0]['instr_encoding'],
                    'path': trajectory,
                    'observations': path_observations,
                    'actions': path_actions,
                    'score': inf_state.score,
                    'scores': path_scores,
                    'attentions': path_attentions
                })
            trajs.append(this_trajs)
        return trajs, completed_list, traversed_lists


    def beam_search(self, beam_size):
        assert self.decoder.panoramic
        # assert self.env.beam_size >= beam_size
        world_states = self.env.reset(True)  # [(feature, state)]
        obs = np.array(self.env._get_obs(world_states))
        batch_size = len(world_states)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch([o for ob in obs for o in ob])

        world_states = [[world_state for f, world_state in states] for states in world_states]

        ctx, h_t, c_t, seq_mask = self.encoder(seq, seq_mask, seq_lengths)
        if not self.decoder_init:
            h_t, c_t = torch.zeros_like(h_t), torch.zeros_like(c_t)

        completed = []
        for _ in range(batch_size):
            completed.append([])

        beams = [[InferenceState(prev_inference_state=None,
                                 world_state=ws[0],
                                 observation=o[0],
                                 flat_index=i,
                                 last_action=-1,
                                 last_action_embedding=self.decoder.u_begin,
                                 action_count=0,
                                 score=0.0,
                                 h_t=None, c_t=None,
                                 last_alpha=None)]
                 for i, (ws, o) in enumerate(zip(world_states, obs))]
        #
        # batch_size x beam_size

        for t in range(self.episode_len):
            flat_indices = []
            beam_indices = []
            u_t_list = []
            for beam_index, beam in enumerate(beams):
                for inf_state in beam:
                    beam_indices.append(beam_index)
                    flat_indices.append(inf_state.flat_index)
                    u_t_list.append(inf_state.last_action_embedding)

            u_t_prev = torch.stack(u_t_list, dim=0)
            flat_obs = [ob for obs_beam in obs for ob in obs_beam]

            # Image features from obs
            # if self.decoder.panoramic:
            f_t_all, f_t = self._feature_variable(flat_obs)


            # Action feature from obs
            # if self.decoder.action_space == 6:
            #     u_t_features, is_valid = np.zeros((batch_size, 1)), None
            # else:
            u_t_features, is_valid, is_valid_numpy = self._action_variable(flat_obs)

            h_t, c_t, alpha, logit, pred_f = self.decoder(None, u_t_prev, u_t_features,
                                                          f_t, f_t_all, h_t[flat_indices], c_t[flat_indices],
                  [ctx_si[beam_indices] for ctx_si in ctx] if isinstance(ctx, list) else ctx[beam_indices],
                  [seq_mask_si[beam_indices] for seq_mask_si in seq_mask] if isinstance(ctx, list) else seq_mask[beam_indices])
            # Mask outputs where agent can't move forward

            logit[is_valid == 0] = -float('inf')

            masked_logit = logit  # for debug
            log_probs = F.log_softmax(logit, dim=1).data

            _, action_indices = masked_logit.data.topk(min(beam_size, logit.size()[1]), dim=1)
            action_scores = log_probs.gather(1, action_indices)
            assert action_scores.size() == action_indices.size()

            start_index = 0
            new_beams = []
            assert len(beams) == len(world_states)
            all_successors = []
            for beam_index, (beam, beam_world_states, beam_obs) in enumerate(zip(beams, world_states, obs)):
                successors = []
                end_index = start_index + len(beam)
                assert len(beam_obs) == len(beam) and len(beam_world_states) == len(beam)
                if beam:
                    for inf_index, (inf_state, world_state, ob, action_score_row, action_index_row) in \
                            enumerate(zip(beam, beam_world_states, beam_obs, action_scores[start_index:end_index],
                                          action_indices[start_index:end_index])):
                        flat_index = start_index + inf_index
                        for action_score, action_index in zip(action_score_row, action_index_row):
                            if is_valid_numpy[flat_index, action_index] == 0:
                                continue
                            successors.append(
                                InferenceState(prev_inference_state=inf_state,
                                   world_state=world_state,
                                   # will be updated later after successors are pruned
                                   observation=ob,  # will be updated later after successors are pruned
                                   flat_index=flat_index,
                                   last_action=action_index,
                                   last_action_embedding=u_t_features[flat_index, action_index].detach(),
                                   action_count=inf_state.action_count + 1,
                                   score=float(inf_state.score + action_score), h_t=None, c_t=None,
                                   last_alpha=[alpha_si[flat_index].data for alpha_si in alpha] if isinstance(alpha, list) else alpha[flat_index].data)
                            )
                start_index = end_index
                successors = sorted(successors, key=lambda t: t.score, reverse=True)[:beam_size]
                all_successors.append(successors)

            successor_world_states = [
                [inf_state.world_state for inf_state in successors]
                for successors in all_successors
            ]

            successor_env_actions = [
                [inf_state.last_action for inf_state in successors]
                for successors in all_successors
            ]

            successor_last_obs = [
                [inf_state.observation for inf_state in successors]
                for successors in all_successors
            ]

            successor_world_states=self.env.step(successor_env_actions, successor_last_obs, successor_world_states)
            successor_obs = np.array(self.env._get_obs(successor_world_states))
            successor_world_states = [[world_state for f, world_state in states] for states in successor_world_states]

            acc = []
            for ttt in zip(all_successors, successor_world_states, successor_obs):
                mapped = [inf._replace(world_state=ws, observation=o) for inf, ws, o in zip(*ttt)]
                acc.append(mapped)
            all_successors=acc


            for beam_index, successors in enumerate(all_successors):
                new_beam = []
                for successor in successors:
                    if successor.last_action == 0 or t == self.episode_len - 1:
                        completed[beam_index].append(successor)
                    else:
                        new_beam.append(successor)
                if len(completed[beam_index]) >= beam_size:
                    new_beam = []
                new_beams.append(new_beam)

            beams = new_beams

            world_states = [
                [inf_state.world_state for inf_state in beam]
                for beam in beams
            ]

            obs = [
                [inf_state.observation for inf_state in beam]
                for beam in beams
            ]

            # Early exit if all ended
            if not any(beam for beam in beams):
                break

        trajs = []

        for this_completed in completed:
            assert this_completed
            this_trajs = []
            for inf_state in sorted(this_completed, key=lambda t: t.score, reverse=True)[:beam_size]:
                path_states, path_observations, path_actions, path_scores, path_attentions = backchain_inference_states(inf_state)
                # this will have messed-up headings for (at least some) starting locations because of
                # discretization, so read from the observations instead
                ## path = [(obs.viewpointId, state.heading, state.elevation)
                ##         for state in path_states]
                trajectory = [path_element_from_observation(ob) for ob in path_observations]
                this_trajs.append({
                    'instr_id': path_observations[0]['instr_id'],
                    'instr_encoding': path_observations[0]['instr_encoding'],
                    'path': trajectory,
                    'observations': path_observations,
                    'actions': path_actions,
                    'score': inf_state.score,
                    'scores': path_scores,
                    'attentions': path_attentions
                })
            trajs.append(this_trajs)
        traversed_lists = None  # todo
        return trajs, completed, traversed_lists


    def rollout_with_loss(self):
        world_states = self.env.reset(False)
        obs = np.array(self.env._get_obs(world_states))
        batch_size = len(obs)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]
        # when there are multiple sentences, perm_idx will simply be range(batch_size).
        # but perm_idx for each batch of i-th(i=1 or 2 or 3) will be inside seq_length.
        # this means:
        # seq_lengths=[(seq_lengths,perm_idx),(seq_lengths,perm_idx),(seq_lengths,perm_idx)]

        # Record starting point
        traj = [{'instr_id': ob['instr_id'], 'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]} for ob in perm_obs]

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx, h_t, c_t, seq_mask = self.encoder(seq, seq_mask, seq_lengths)

        if not self.decoder_init:
            #h_t, c_t = torch.zeros_like(h_t), torch.zeros_like(c_t)
            c_t = torch.zeros_like(c_t) # debug

        # Initial action
        a_t_prev, u_t_prev = None, None  # different action space
        if self.decoder.action_space == -1:
            u_t_prev = self.decoder.u_begin.expand(batch_size, -1)
        else:
            a_t_prev = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'), requires_grad=False).to(device)

        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Do a sequence rollout and calculate the loss
        self.loss = 0
        self.loss_ctrl_f = 0

        env_action = [None] * batch_size

        # for plot
        #all_alpha = []
        action_scores = np.zeros((batch_size,))

        for t in range(self.episode_len):
            f_t = self._feature_variable(perm_obs)

            # Image features from obs
            if self.decoder.panoramic: f_t_all, f_t = f_t
            else: f_t_all = np.zeros((batch_size, 1))

            # Action feature from obs
            if self.decoder.action_space == 6:
                u_t_features, is_valid = np.zeros((batch_size, 1)), None
            else:
                u_t_features, is_valid, _ = self._action_variable(perm_obs)

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            h_t, c_t, alpha, logit, pred_f = self.decoder(a_t_prev, u_t_prev, u_t_features, f_t, f_t_all, h_t, c_t, ctx, seq_mask)

            # all_alpha.append(alpha)
            # Mask outputs where agent can't move forward
            if self.decoder.action_space == 6:
                for i,ob in enumerate(perm_obs):
                    if len(ob['navigableLocations']) <= 1:
                        logit[i, self.model_actions.index('forward')] = -float('inf')
            else:
                logit[is_valid == 0] = -float('inf')
                # if self.decoder.ctrl_feature:
                #     pred_f[is_valid == 0] = 0

            if self.temp_alpha != 0: # add temperature
                logit = logit/self.temp_alpha

            self.loss += self.criterion(logit, target)

            # Auxiliary training
            if self.decoder.ctrl_feature:
                target_f = self._teacher_feature(perm_obs, ended)#, is_valid.shape[-1])
                self.loss_ctrl_f += self.aux_ratio * self.criterion_ctrl_f(pred_f, target_f)
            # todo: add auxiliary tasks to sc-rl training?

            # Determine next model inputs
            # scheduled sampling
            if self.schedule_ratio >= 0 and self.schedule_ratio <= 1:
                sample_feedback = random.choices(['sample', 'teacher'], [self.schedule_ratio, 1 - self.schedule_ratio], k=1)[0]  # schedule sampling
                if self.feedback != 'argmax': # ignore test case
                    self.feedback = sample_feedback

            a_t = self.get_next(self.feedback, target, logit)

            # setting A
            if self.testing_settingA:
                log_probs = F.log_softmax(logit, dim=1).data
                action_score = log_probs[torch.arange(batch_size), a_t].cpu().data.numpy()

            # log_probs = F.log_softmax(logit, dim=1).data
            # action_score = log_probs[torch.arange(batch_size), a_t].cpu().data.numpy()

            a_t_prev = a_t
            if self.decoder.action_space != 6:  # update the previous action
                u_t_prev = u_t_features[np.arange(batch_size), a_t, :].detach()

            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                if self.decoder.action_space == 6:
                    if action_idx == self.model_actions.index('<end>'):
                    #     # sub_stages[idx] = max(sub_stages[idx]-1, 0)
                    #     # ended[i] = (sub_stages[idx]==0)
                        ended[i] = True
                    env_action[idx] = self.env_actions[action_idx]
                else:
                    if action_idx == 0:
                    #     # sub_stages[idx] = max(sub_stages[idx] - 1, 0)
                    #     # ended[i] = (sub_stages[idx] == 0)
                        ended[i] = True
                    env_action[idx] = action_idx

            # state transitions
            new_states = self.env.step(env_action, obs)
            obs = np.array(self.env._get_obs(new_states))
            #obs = np.array(self.env._get_obs(self.env.step(env_action, obs)))  # , sub_stages
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()

                if self.testing_settingA:
                    if not ended[i]:
                        action_scores[idx] = action_scores[idx] + action_score[i]

                if self.decoder.action_space == 6:
                    if action_idx == self.model_actions.index('<end>'):
                        ended[i] = True
                else:
                    if action_idx == 0:
                        ended[i] = True

            # for i,idx in enumerate(perm_idx):
            #     action_idx = a_t[i].item()
            #     # if not ended[i]:
            #     #     action_scores[idx] = action_scores[idx] + action_score[i]
            #     if self.decoder.action_space == 6:
            #         if action_idx == self.model_actions.index('<end>'):
            #             ended[i] = True
            #     else:
            #         if action_idx == 0:
            #             ended[i] = True

            # Early exit if all ended
            if ended.all(): break

        # episode_len is just a constant so it doesn't matter
        self.losses.append(self.loss.item())  # / self.episode_len)
        if self.decoder.ctrl_feature:
            self.losses_ctrl_f.append(self.loss_ctrl_f.item())  # / self.episode_len)

        # with open('preprocess/alpha.pkl', 'wb') as alpha_f:  # TODO: remove for release!!!!
        #     pickle.dump(all_alpha, alpha_f)

        # chooes one from three according to prob
        # for t,p in zip(traj, action_scores):
        #     t['prob'] = p

        # chooes one from three according to prob
        if self.testing_settingA:
            for t, p in zip(traj, action_scores):
                t['prob'] = p

        return traj


    def pretrain_rollout_with_loss(self):
        world_states = self.env.reset(False)
        obs = np.array(self.env._get_obs(world_states))
        batch_size = len(obs)

        # Reorder the language input for the encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]
        # when there are multiple sentences, perm_idx will simply be range(batch_size).
        # but perm_idx for each batch of i-th(i=1 or 2 or 3) will be inside seq_length.
        # this means:
        # seq_lengths=[(seq_lengths,perm_idx),(seq_lengths,perm_idx),(seq_lengths,perm_idx)]

        # Record starting point
        traj = [{'instr_id': ob['instr_id'], 'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]} for ob in perm_obs]

        ## Forward through encoder, giving initial hidden state and memory cell for decoder
        #f_t = self._feature_variable(perm_obs)
        #if self.decoder.panoramic: f_t_all, f_t = f_t
        #else: f_t_all = np.zeros((batch_size, 1))
        #ctx, h_t, c_t, seq_mask = self.encoder(seq, seq_mask, seq_lengths,f_t_all=f_t_all)
        ###ctx, en_ht, en_ct, vl_mask = self.encoder(seq, seq_mask, seq_lengths,f_t_all=f_t_all)
        #if not self.decoder_init:
        #    h_t, c_t = torch.zeros_like(h_t), torch.zeros_like(c_t)
        #    #c_t = torch.zeros_like(c_t) # debug

        # Initial action
        a_t_prev, u_t_prev = None, None  # different action space
        if self.decoder.action_space == -1:
            u_t_prev = self.decoder.u_begin.expand(batch_size, -1)
        else:
            a_t_prev = Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'), requires_grad=False).to(device)

        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Do a sequence rollout and calculate the loss
        self.loss = 0
        self.loss_ctrl_f = 0

        env_action = [None] * batch_size

        # for plot
        #all_alpha = []
        action_scores = np.zeros((batch_size,))
        for t in range(self.episode_len):
            f_t = self._feature_variable(perm_obs)

            # Image features from obs
            if self.decoder.panoramic: f_t_all, f_t = f_t
            else: f_t_all = np.zeros((batch_size, 1))

            # Action feature from obs
            if self.decoder.action_space == 6:
                u_t_features, is_valid = np.zeros((batch_size, 1)), None
            else:
                u_t_features, is_valid, _ = self._action_variable(perm_obs)

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            ctx, en_ht, en_ct, vl_mask = self.encoder(seq, seq_mask, seq_lengths, f_t_all=f_t_all)
            if t == 0: # use encoder's ht and ct as init
                de_ht, de_ct, alpha, logit, pred_f = self.decoder(a_t_prev, u_t_prev, u_t_features, f_t, f_t_all, en_ht, en_ct, ctx, vl_mask)
            else:  # otherwise unroll as lstm
                de_ht, de_ct, alpha, logit, pred_f = self.decoder(a_t_prev, u_t_prev, u_t_features, f_t, f_t_all, de_ht, de_ct, ctx, vl_mask)


            # all_alpha.append(alpha)
            # Mask outputs where agent can't move forward
            if self.decoder.action_space == 6:
                for i,ob in enumerate(perm_obs):
                    if len(ob['navigableLocations']) <= 1:
                        logit[i, self.model_actions.index('forward')] = -float('inf')
            else:
                logit[is_valid == 0] = -float('inf')
                # if self.decoder.ctrl_feature:
                #     pred_f[is_valid == 0] = 0

            if self.temp_alpha != 0: # add temperature
                logit = logit/self.temp_alpha

            self.loss += self.criterion(logit, target)

            # Auxiliary training
            if self.decoder.ctrl_feature:
                target_f = self._teacher_feature(perm_obs, ended)#, is_valid.shape[-1])
                self.loss_ctrl_f += self.aux_ratio * self.criterion_ctrl_f(pred_f, target_f)
            # todo: add auxiliary tasks to sc-rl training?

            # Determine next model inputs
            # scheduled sampling
            if self.schedule_ratio >= 0 and self.schedule_ratio <= 1:
                sample_feedback = random.choices(['sample', 'teacher'], [self.schedule_ratio, 1 - self.schedule_ratio], k=1)[0]  # schedule sampling
                if self.feedback != 'argmax': # ignore test case
                    self.feedback = sample_feedback

            a_t = self.get_next(self.feedback, target, logit)

            # setting A
            if self.testing_settingA:
                log_probs = F.log_softmax(logit, dim=1).data
                action_score = log_probs[torch.arange(batch_size), a_t].cpu().data.numpy()

            # log_probs = F.log_softmax(logit, dim=1).data
            # action_score = log_probs[torch.arange(batch_size), a_t].cpu().data.numpy()

            a_t_prev = a_t
            if self.decoder.action_space != 6:  # update the previous action
                u_t_prev = u_t_features[np.arange(batch_size), a_t, :].detach()

            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                if self.decoder.action_space == 6:
                    if action_idx == self.model_actions.index('<end>'):
                    #     # sub_stages[idx] = max(sub_stages[idx]-1, 0)
                    #     # ended[i] = (sub_stages[idx]==0)
                        ended[i] = True
                    env_action[idx] = self.env_actions[action_idx]
                else:
                    if action_idx == 0:
                    #     # sub_stages[idx] = max(sub_stages[idx] - 1, 0)
                    #     # ended[i] = (sub_stages[idx] == 0)
                        ended[i] = True
                    env_action[idx] = action_idx

            # state transitions
            new_states = self.env.step(env_action, obs)
            obs = np.array(self.env._get_obs(new_states))
            #obs = np.array(self.env._get_obs(self.env.step(env_action, obs)))  # , sub_stages
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))

            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()

                if self.testing_settingA:
                    if not ended[i]:
                        action_scores[idx] = action_scores[idx] + action_score[i]

                if self.decoder.action_space == 6:
                    if action_idx == self.model_actions.index('<end>'):
                        ended[i] = True
                else:
                    if action_idx == 0:
                        ended[i] = True

            # for i,idx in enumerate(perm_idx):
            #     action_idx = a_t[i].item()
            #     # if not ended[i]:
            #     #     action_scores[idx] = action_scores[idx] + action_score[i]
            #     if self.decoder.action_space == 6:
            #         if action_idx == self.model_actions.index('<end>'):
            #             ended[i] = True
            #     else:
            #         if action_idx == 0:
            #             ended[i] = True

            # Early exit if all ended
            if ended.all(): break

        # episode_len is just a constant so it doesn't matter
        self.losses.append(self.loss.item())  # / self.episode_len)
        if self.decoder.ctrl_feature:
            self.losses_ctrl_f.append(self.loss_ctrl_f.item())  # / self.episode_len)

        # with open('preprocess/alpha.pkl', 'wb') as alpha_f:  # TODO: remove for release!!!!
        #     pickle.dump(all_alpha, alpha_f)

        # chooes one from three according to prob
        # for t,p in zip(traj, action_scores):
        #     t['prob'] = p

        # chooes one from three according to prob
        if self.testing_settingA:
            for t, p in zip(traj, action_scores):
                t['prob'] = p

        return traj



    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, beam_size=1, successors=1, speaker=(None,None,None,None)):
        ''' Evaluate once on each instruction in the current environment '''
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        super(Seq2SeqAgent, self).test(beam_size, successors, speaker)

    def train(self, encoder_optimizer, decoder_optimizer, n_iters, aux_n_iter, feedback='teacher'):
        ''' Train for a given number of iterations '''
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        self.losses_ctrl_f = []
        epo_inc = 0
        for iter in range(1, n_iters + 1):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            self.rollout()
            if (not self.decoder.ctrl_feature) or (iter % aux_n_iter):
                self.loss.backward()
            else:
                self.loss_ctrl_f.backward()

            if self.clip_gradient != 0: # clip gradient
                clip_gradient(encoder_optimizer, self.clip_gradient)
                clip_gradient(decoder_optimizer, self.clip_gradient)

            if self.clip_gradient_norm > 0: # clip gradient norm
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_gradient_norm)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip_gradient_norm)

            encoder_optimizer.step()
            decoder_optimizer.step()
            epo_inc += self.env.epo_inc
        return epo_inc

    def rollout_notrain(self, n_iters):  # jolin
        epo_inc = 0
        for iter in range(1, n_iters + 1):
            self.env._next_minibatch(False)
            epo_inc += self.env.epo_inc
        return epo_inc

    def rl_rollout(self, obs, perm_obs, seq, seq_mask, seq_lengths, perm_idx, feedback,
                   encoder, decoder):
        batch_size = len(perm_obs)
        # Record starting point
        traj = [{'instr_id': ob['instr_id'],
                 'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx, h_t, c_t, seq_mask = encoder(seq, seq_mask, seq_lengths)
        if not self.decoder_init:
            h_t, c_t = torch.zeros_like(h_t), torch.zeros_like(c_t)

        # Initial action
        a_t_prev, u_t_prev = None, None  # different action space
        if decoder.action_space==-1:
            u_t_prev = decoder.u_begin.expand(batch_size, -1)
        else:
            a_t_prev = Variable(torch.ones(batch_size).long() *
                self.model_actions.index('<start>'), requires_grad=False).to(device)

        ended=np.array([False] * batch_size)

        # Do a sequence rollout not don't calculate the loss for policy gradient
        # self.loss = 0
        env_action = [None] * batch_size

        # Initialize seq log probs for policy gradient
        if feedback == 'sample1':
            seqLogprobs = h_t.new_zeros(batch_size, self.episode_len)
            mask = np.ones((batch_size, self.episode_len))
        elif feedback == 'argmax1':
            seqLogprobs, mask = None, None
        else:
            raise NotImplementedError('other feedback not supported.')

        # only for supervised auxiliary tasks
        #assert (not self.decoder.ctrl_feature)  # not implemented
        for t in range(self.episode_len):
            f_t = self._feature_variable(perm_obs)

            # Image features from obs
            if decoder.panoramic: f_t_all, f_t = f_t
            else: f_t_all = np.zeros((batch_size, 1))

            # Action feature from obs
            if decoder.action_space == 6:
                u_t_features, is_valid = np.zeros((batch_size, 1)), None
            else:
                u_t_features, is_valid, _ = self._action_variable(perm_obs)

            # Decoding actions
            h_t, c_t, alpha, logit, pred_f = decoder(a_t_prev, u_t_prev, u_t_features,
                                               f_t, f_t_all, h_t, c_t, ctx, seq_mask)

            # Mask outputs where agent can't move forward
            if decoder.action_space == 6:
                for i,ob in enumerate(perm_obs):
                    if len(ob['navigableLocations']) <= 1:
                        logit[i, self.model_actions.index('forward')] = -float('inf')
            else: logit[is_valid == 0] = -float('inf')

            # Supervised training
            # target = self._teacher_action(perm_obs, ended)
            # self.loss += self.criterion(logit, target)

            # Determine next model inputs
            if feedback == 'argmax1':
                _, a_t = logit.max(1)
            elif feedback == 'sample1':
                logprobs = F.log_softmax(logit, dim=1)
                probs = torch.exp(logprobs.data)
                m = D.Categorical(probs)
                a_t = m.sample()  # sampling an action from model
                sampleLogprobs = logprobs.gather(1, a_t.unsqueeze(1))
            else:
                sys.exit('invalid feedback method %s'%feedback)
            # if self.decoder.panoramic:
            #     a_t_feature = all_a_t_features[np.arange(batch_size), a_t, :].detach()

            # Updated 'ended' list and make environment action
            for i,idx in enumerate(perm_idx):
                action_idx = a_t[i].item()
                if decoder.action_space == 6:
                    if ended[i] and mask is not None:
                            mask[i, t] = 0
                    elif action_idx == self.model_actions.index('<end>'):
                        ended[i] = True
                    env_action[idx] = self.env_actions[action_idx]
                else:
                    if ended[i] and mask is not None:
                            mask[i, t] = 0
                    elif action_idx == 0: ended[i] = True
                    env_action[idx] = action_idx

            obs = np.array(self.env._get_obs(self.env.step(env_action, obs)))
            perm_obs = obs[perm_idx]

            # Save trajectory output
            for i,ob in enumerate(perm_obs):
                if not ended[i]:
                    traj[i]['path'].append((ob['viewpoint'],
                                            ob['heading'], ob['elevation']))

            if seqLogprobs is not None:  # feedback == 'sample1'
                seqLogprobs[:, t] = sampleLogprobs.view(-1)

            # Early exit if all ended
            if ended.all(): break

        path_res = {}
        for t in traj:
            path_res[t['instr_id']] = t['path']
        return traj, mask, seqLogprobs, path_res

    def rl_train(self, train_Eval, encoder_optimizer, decoder_optimizer,
                 n_iters, sc_reward_scale, sc_discouted_immediate_r_scale, sc_length_scale):
        ''' jolin: self-critical finetuning'''
        self.losses = []
        epo_inc = 0
        self.encoder.train()
        self.decoder.train()
        for iter in range(1, n_iters + 1):  # n_iters=interval
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            # copy from self.rollout():
            # one minibatch (100 instructions)
            world_states = self.env.reset(False)
            obs = np.array(self.env._get_obs(world_states))
            epo_inc += self.env.epo_inc
            # Reorder the language input for the encoder
            seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
            perm_obs = obs[perm_idx]
            gen_traj, mask, seqLogprobs, gen_results = self.rl_rollout(obs, perm_obs, seq, seq_mask,
                                                             seq_lengths, perm_idx, 'sample1',
                                                             self.encoder, self.decoder)

            # jolin: get greedy decoding baseline
            # Just like self.test(use_dropout=False, feedback='argmax').
            # But we should not do env.reset_epoch(), because we do not
            # test the whole split. So DON'T reuse test()!

            world_states = self.env.reset_batch()
            obs = np.array(self.env._get_obs(world_states))# for later 'sample' feedback batch
            perm_obs = obs[perm_idx]

            if self.monotonic:
                encoder2, decoder2 = self.encoder2, self.decoder2
            else:
                self.encoder.eval()
                self.decoder.eval()
                encoder2, decoder2 = self.encoder, self.decoder
            with torch.no_grad():
                greedy_traj, _, _, greedy_res = self.rl_rollout(obs, perm_obs, seq, seq_mask,
                                                      seq_lengths, perm_idx, 'argmax1',
                                                      encoder2, decoder2)
            if not self.monotonic:
                self.encoder.train()
                self.decoder.train()

            # jolin: get self-critical reward
            reward = self.get_self_critical_reward(gen_traj, train_Eval, gen_results, greedy_res, sc_reward_scale, sc_discouted_immediate_r_scale, sc_length_scale)

            # jolin: RewardCriterion
            self.loss = self.PG_reward_criterion(seqLogprobs, reward, mask)

            self.losses.append(self.loss.item())
            self.loss.backward()
            #clip_gradient(encoder_optimizer)
            #clip_gradient(decoder_optimizer)

            if self.clip_gradient != 0: # clip gradient
                clip_gradient(encoder_optimizer, self.clip_gradient)
                clip_gradient(decoder_optimizer, self.clip_gradient)

            if self.clip_gradient_norm > 0: # clip gradient norm
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_gradient_norm)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip_gradient_norm)

            encoder_optimizer.step()
            decoder_optimizer.step()
        return epo_inc

    def PG_reward_criterion(self, seqLogprobs, reward, mask):
        # jolin: RewardCriterion
        input = to_contiguous(seqLogprobs).view(-1)
        reward = to_contiguous(torch.from_numpy(reward).float().to(device)).view(-1)
        #mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        mask = to_contiguous(torch.from_numpy(mask).float().to(device)).view(-1)
        output = - input * reward * mask
        loss = torch.sum(output) / torch.sum(mask)
        return loss

    def get_self_critical_reward(self, traj, Eval, gen_results, greedy_res, sc_reward_scale, sc_discouted_immediate_r_scale, sc_length_scale):
        # get self-critical reward
        instr_id_order = [t['instr_id'] for t in traj]
        gen_scores = Eval.score_batch(gen_results, instr_id_order)
        greedy_scores = Eval.score_batch(greedy_res, instr_id_order)

        # normal score
        gen_hits = (np.array(gen_scores['nav_errors']) <= 3.0).astype(float)
        greedy_hits = (np.array(greedy_scores['nav_errors']) <= 3.0).astype(float)
        gen_lengths = (np.array(gen_scores['trajectory_lengths'])).astype(float)

        # sr_sc
        hits = gen_hits - greedy_hits
        #reward = np.repeat(hits[:, np.newaxis], self.episode_len, 1)*sc_reward_scale

        # spl
        gen_spls = (np.array(gen_scores['spl'])).astype(float)
        greedy_spls = (np.array(greedy_scores['spl'])).astype(float)

        ave_steps = (np.array(gen_scores['trajectory_steps'])).sum()/float(len(instr_id_order))
        steps = (np.array(gen_scores['trajectory_steps']) - self.episode_len).sum()

        if self.reward_func == 'sr_sc':
            reward = np.repeat(hits[:, np.newaxis], self.episode_len, 1)*sc_reward_scale
        elif self.reward_func == 'spl':
            reward = np.repeat(gen_spls[:, np.newaxis], self.episode_len, 1) * sc_reward_scale
        elif self.reward_func == 'spl_sc':
            # spl_sc
            spl_sc = gen_spls - greedy_spls
            reward = np.repeat(spl_sc[:, np.newaxis], self.episode_len, 1) * sc_reward_scale
        elif self.reward_func == 'spl_last': # does not work
            tj_steps = [s - 1 for s in gen_scores['trajectory_steps']] # tj_steps
            reward = np.zeros((gen_spls.shape[0], self.episode_len), dtype=float)
            reward[range(gen_spls.shape[0]), tj_steps] = gen_scores['spl']
        elif self.reward_func == 'spl_last_sc':
            tj_steps = [s - 1 for s in gen_scores['trajectory_steps']] # tj_steps
            reward = np.zeros((gen_spls.shape[0], self.episode_len), dtype=float)
            reward[range(gen_spls.shape[0]), tj_steps] = [x - y for x, y in zip(gen_scores['spl'], greedy_scores['spl'])]
        elif self.reward_func == 'spl_psc': # test
            tj_steps = [s - 1 for s in gen_scores['trajectory_steps']]  # tj_steps
            reward = np.full((gen_spls.shape[0], self.episode_len), -sc_length_scale, dtype=float) # penalty
            reward[range(gen_spls.shape[0]), tj_steps] = gen_scores['spl']

        # discounted immediate reward
        if sc_discouted_immediate_r_scale>0:
            discounted_r = discount_rewards(gen_scores['immediate_rewards'], self.episode_len) * sc_discouted_immediate_r_scale
            reward = reward + discounted_r

        # panelty for length
        if sc_length_scale:
            length_panelty = np.repeat(gen_lengths[:, np.newaxis], self.episode_len, 1)*sc_length_scale
            reward = reward - length_panelty
        return reward


    def save(self, encoder_path, decoder_path):
        ''' Snapshot models '''

        write_num = 0
        while (write_num < 10):
            try:
                torch.save(self.encoder.state_dict(), encoder_path)
                torch.save(self.decoder.state_dict(), decoder_path)
                if torch.cuda.is_available():
                    torch.save(torch.cuda.random.get_rng_state(), decoder_path + '.rng.gpu')
                torch.save(torch.random.get_rng_state(), decoder_path + '.rng')
                with open(decoder_path + '.rng2', 'wb') as f:
                    pickle.dump(random.getstate(), f)
                break
            except:
                write_num += 1

    def delete(self, encoder_path, decoder_path):
        ''' Delete models '''
        os.remove(encoder_path)
        os.remove(decoder_path)
        os.remove(decoder_path+'.rng.gpu')
        os.remove(decoder_path+'.rng')
        os.remove(decoder_path+'.rng2')

    def load(self, encoder_path, decoder_path):
        ''' Loads parameters (but not training state) '''
        self.encoder.load_state_dict(torch.load(encoder_path, 'cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.decoder.load_state_dict(torch.load(decoder_path, 'cuda:0' if torch.cuda.is_available() else 'cpu'), strict=False)
        self.encoder.to(device)
        self.decoder.to(device)
        if self.monotonic:
            self.copy_seq2seq()

        try:
            with open(decoder_path+'.rng2','rb') as f:
                random.setstate(pickle.load(f))
            torch.random.set_rng_state(torch.load(decoder_path + '.rng'))
            torch.cuda.random.set_rng_state(torch.load(decoder_path + '.rng.gpu'))
        except FileNotFoundError:
            print('Warning: failed to find random seed file')

    def copy_seq2seq(self):
        self.encoder2=copy.deepcopy(self.encoder)
        self.decoder2=copy.deepcopy(self.decoder)
        self.encoder2.eval()
        self.decoder2.eval()
        for param in self.encoder2.parameters():
            param.requires_grad = False
        for param in self.decoder2.parameters():
            param.requires_grad = False

