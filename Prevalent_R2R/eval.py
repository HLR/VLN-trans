''' Evaluation of agent trajectories '''

import json
import os
import sys
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from env import R2RBatch
from utils import load_datasets, load_nav_graphs
from agent import BaseAgent, StopAgent, RandomAgent, ShortestAgent
from feature import Feature


class Evaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits, encoder_type):#, subgoal):
        self.error_margin = 3.0
        self.splits = splits
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        for item in load_datasets(splits, 'lstm'):#, subgoal):# no matter what encoder type
            self.gt[item['path_id']] = item
            self.scans.append(item['scan'])
            self.instr_ids += ['%d_%d' % (item['path_id'],i) for i in range(3)]
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, path):
        ''' Calculate error based on the final position in trajectory, and also 
            the closest position (oracle stopping rule). '''
        gt = self.gt[int(instr_id.split('_')[0])]
        start = gt['path'][0]
        assert start == path[0][0], \
            'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        nav_error = self.distances[gt['scan']][final_position][goal]
        oracle_error = self.distances[gt['scan']][nearest_position][goal]
        trajectory_steps = len(path) - 1
        trajectory_length = 0  # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            trajectory_length += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        success = nav_error < self.error_margin
        # check for type errors
        # assert success == True or success == False
        # check for type errors
        oracle_success = oracle_error < self.error_margin
        # assert oracle_success == True or oracle_success == False
        sp_length = self.distances[gt['scan']][gt['path'][0]][gt['path'][-1]]
        spl = 0.0 if nav_error >= self.error_margin else \
            (float(sp_length) / max(trajectory_length, sp_length))

        self.scores['nav_errors'].append(nav_error)
        self.scores['oracle_errors'].append(oracle_error)
        self.scores['trajectory_steps'].append(trajectory_steps)
        self.scores['trajectory_lengths'].append(trajectory_length)
        self.scores['success'].append(success)
        self.scores['oracle_success'].append(oracle_success)
        self.scores['spl'].append(spl)


    def score_output(self, items):
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        for item in items:
            # Check against expected ids
            if item['instr_id'] in instr_ids:
                instr_ids.remove(item['instr_id'])
                self._score_item(item['instr_id'], item['trajectory'])
        assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s' \
                                    % (len(instr_ids), len(self.instr_ids), ",".join(self.splits))
        assert len(self.scores['nav_errors']) == len(self.instr_ids)

        # success_steps=[steps for i, steps in enumerate(self.scores['trajectory_steps'])
        #                       if self.scores['success'][i]]
        score_summary = {
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle_error': np.average(self.scores['oracle_errors']),
            'steps': np.average(self.scores['trajectory_steps']),
            'lengths': np.average(self.scores['trajectory_lengths']),
            'success_rate': float(
                sum(self.scores['success']) / len(self.scores['success'])),
            'oracle_rate': float(sum(self.scores['oracle_success'])
                                 / len(self.scores['oracle_success'])),
            'spl': float(sum(self.scores['spl'])) / len(self.scores['spl']),
            # 'success_steps': np.average(success_steps),
            # 'failed_steps': float(sum(self.scores['trajectory_steps'])-sum(success_steps))/
            #                (len(self.scores['success'])-len(success_steps)+1e-6)
        }
        return score_summary, self.scores


    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        with open(output_file) as f:
            items = json.load(f)
        return self.score_output(items)


    def score_batch(self, batch, instr_id_order):  # jolin
        #output = [{'instr_id': k, 'trajectory': v} for k, v in self.batch.items()]
        # with open(self.results_path, 'w') as f:
        #     json.dump(output, f)
        scores = defaultdict(list)
        # instr_ids = set(self.instr_ids)
        # with open(output_file) as f:
        #     for item in json.load(f):
        for instr_id in instr_id_order:
            path = batch[instr_id]
            gt = self.gt[int(instr_id.split('_')[0])]
            start = gt['path'][0]
            assert start == path[0][0], \
                'Result trajectories should include the start position'
            goal = gt['path'][-1]
            final_position = path[-1][0]
            nearest_position = self._get_nearest(gt['scan'], goal, path)
            nav_error = self.distances[gt['scan']][final_position][goal]
            oracle_error = self.distances[gt['scan']][nearest_position][goal]
            trajectory_steps = len(path)-1
            trajectory_length = 0  # Work out the length of the path in meters
            prev = path[0]
            for curr in path[1:]:
                trajectory_length += self.distances[gt['scan']][prev[0]][curr[0]]
                prev = curr
            success = nav_error < self.error_margin
            oracle_success = oracle_error < self.error_margin
            sp_length = 0
            prev = gt['path'][0]
            sp_length = self.distances[gt['scan']][gt['path'][0]][gt['path'][-1]]
            spl = 0.0 if nav_error >= self.error_margin else \
                (float(sp_length) / max(trajectory_length, sp_length))

            scores['nav_errors'].append(nav_error)
            scores['oracle_errors'].append(oracle_error)
            scores['trajectory_steps'].append(trajectory_steps)
            scores['trajectory_lengths'].append(trajectory_length)
            scores['success'].append(success)
            scores['oracle_success'].append(oracle_success)
            scores['spl'].append(spl)
            scores['immediate_rewards'].append([])
            distance_left = [self.distances[gt['scan']][point[0]][goal] for point in path]
            for pi, d in enumerate(distance_left[:-1]):
                scores['immediate_rewards'][-1].append(d - distance_left[pi+1])
        return scores


def eval_simple_agents():
    ''' Run simple baselines on each split. '''
    for split in ['train', 'val_seen', 'val_unseen']:
        env = R2RBatch(Feature(None, False), False, False, 6, False, 'lstm', batch_size=1, splits=[split],tokenizer=None)
        ev = Evaluation([split], encoder_type='lstm')#  subgoal=False)

        for agent_type in ['Stop', 'Shortest', 'Random']:
            outfile = '%s%s_%s_agent.json' % (RESULT_DIR, split, agent_type.lower())
            agent = BaseAgent.get_agent(agent_type)(env, outfile)
            agent.test()
            agent.write_results()
            score_summary, _ = ev.score(outfile)
            print('\n%s' % agent_type)
            pp.pprint(score_summary)


def eval_seq2seq():#  , subgoal=False):
    ''' Eval sequence to sequence models on val splits (iteration selected from training error) '''
    outfiles = [
        #RESULT_DIR + 'seq2seq_teacher_imagenet_%s_iter_5000.json',
        RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_122000.json'
    ]
    for outfile in outfiles:
        for split in ['val_seen', 'val_unseen']:
            ev = Evaluation([split],'lstm') #, subgoal)
            score_summary,scores  = ev.score(outfile % split)
            print('\n%s' % (outfile % split))
            pp.pprint(score_summary)


if __name__ == '__main__':
    RESULT_DIR = 'tasks/R2R/exps/test/'#best122000-sf/su8-b35/'
    #eval_simple_agents()
    eval_seq2seq()

