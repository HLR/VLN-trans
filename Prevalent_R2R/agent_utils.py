import torch
import numpy as np
from utils import padding_idx
from collections import namedtuple


def basic_actions():
    # For now, the agent can't pick which forward move to make - just the one in the middle
    model_actions = ['left', 'right', 'up', 'down', 'forward', '<end>', '<start>', '<ignore>']
    env_actions = [
        (0, -1, 0),  # left
        (0, 1, 0),  # right
        (0, 0, 1),  # up
        (0, 0, -1),  # down
        (1, 0, 0),  # forward
        (0, 0, 0),  # <end>
        (0, 0, 0),  # <start>
        (0, 0, 0)  # <ignore>
    ]
    return model_actions, env_actions


def sort_seq(seq_tensor):
    seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
    seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]  # Full length

    seq_tensor = torch.from_numpy(seq_tensor)
    seq_lengths = torch.from_numpy(seq_lengths)

    # Sort sequences by lengths
    seq_lengths, perm_idx = seq_lengths.sort(0, True)
    sorted_tensor = seq_tensor[perm_idx]
    mask = (sorted_tensor == padding_idx)[:, :seq_lengths[0]]

    return sorted_tensor, mask, seq_lengths, perm_idx


def sort_batch(obs):
    ''' Extract instructions from a list of observations and sort by descending
        sequence length (to enable PyTorch packing). '''
    if isinstance(obs[0]['instr_encoding'], list): # multiple sentences
        sorted_tensors, masks, seqs_lengths = [],[],[]
        if len(obs)>=3:
            n_sents=min(len(obs[0]['instr_encoding']),len(obs[1]['instr_encoding']),len(obs[2]['instr_encoding']))  # some samples have more than 3 instr
        else:
            n_sents=len(obs[0]['instr_encoding'])
        for si in range(n_sents):
            seq_tensor = np.array([ob['instr_encoding'][si] for ob in obs])
            sorted_tensor, mask, seq_lengths, perm_idx = sort_seq(seq_tensor)
            sorted_tensors.append(sorted_tensor)
            masks.append(mask)
            seqs_lengths.append((seq_lengths, perm_idx))
        return sorted_tensors, masks, seqs_lengths, [i for i in range(len(obs))]
    else:
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        sorted_tensor, mask, seq_lengths, perm_idx = sort_seq(seq_tensor)
        return sorted_tensor, mask, seq_lengths, perm_idx

def discount_rewards(rs, max_len, gamma=0.95):  # jolin
    padded_rs = np.zeros((len(rs), max_len))
    discounted_r = np.zeros((len(rs), max_len))
    for i, r in enumerate(rs):
        padded_rs[i, :len(r)] = r
    running_add = np.zeros(len(rs))
    for t in reversed(range(max_len)):
        running_add = running_add * gamma + padded_rs[:, t]
        discounted_r[:, t] = running_add
    # # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    # for i, r in enumerate(rs):
    #     if len(r):
    #         discounted_r[i, :len(r)] -= np.mean(discounted_r[i, :len(r)])
    #         discounted_r[i, :len(r)] /= (np.std(discounted_r[i, :len(r)])+1e-6)
    return discounted_r


def teacher_action(model_actions, action_space, obs, ended, ignore_index):
    ''' Extract teacher actions into variable. '''
    a = torch.LongTensor(len(obs))
    for i,ob in enumerate(obs):
        # Supervised teacher only moves one axis at a time
        if action_space==6:
            ix,heading_chg,elevation_chg = ob['teacher']
            if heading_chg > 0:
                a[i] = model_actions.index('right')
            elif heading_chg < 0:
                a[i] = model_actions.index('left')
            elif elevation_chg > 0:
                a[i] = model_actions.index('up')
            elif elevation_chg < 0:
                a[i] = model_actions.index('down')
            elif ix > 0:
                a[i] = model_actions.index('forward')
            elif ended[i]:
                a[i] = model_actions.index('<ignore>')
            else:
                a[i] = model_actions.index('<end>')
        else:
            a[i] = ob['teacher'] if not ended[i] else ignore_index
    return a

# for beam_search

InferenceState = namedtuple("InferenceState", "prev_inference_state, world_state, observation, flat_index, last_action, last_action_embedding, action_count, score, h_t, c_t, last_alpha")

WorldState = namedtuple("WorldState", ["scanId", "viewpointId", "heading", "elevation"])


Cons = namedtuple("Cons", "first, rest")


def backchain_inference_states(last_inference_state):
    states = []
    observations = []
    actions = []
    inf_state = last_inference_state
    scores = []
    last_score = None
    attentions = []
    while inf_state is not None:
        states.append(inf_state.world_state)
        observations.append(inf_state.observation)
        actions.append(inf_state.last_action)
        attentions.append(inf_state.last_alpha)
        if last_score is not None:
            scores.append(last_score - inf_state.score)
        last_score = inf_state.score
        inf_state = inf_state.prev_inference_state
    scores.append(last_score)
    return list(reversed(states)), list(reversed(observations)), list(reversed(actions))[1:], list(reversed(scores))[1:], list(reversed(attentions))[1:] # exclude start action


def path_element_from_observation(ob):
    return (ob['viewpoint'], ob['heading'], ob['elevation'])

def cons_to_list(cons):
    l = []
    while True:
        l.append(cons.first)
        cons = cons.rest
        if cons is None:
            break
    return l

def least_common_viewpoint_path(inf_state_a, inf_state_b):
    # return inference states traversing from A to X, then from Y to B,
    # where X and Y are the least common ancestors of A and B respectively that share a viewpointId
    path_to_b_by_viewpoint =  {
    }
    b = inf_state_b
    b_stack = Cons(b, None)
    while b is not None:
        path_to_b_by_viewpoint[b.world_state.viewpointId] = b_stack
        b = b.prev_inference_state
        b_stack = Cons(b, b_stack)
    a = inf_state_a
    path_from_a = [a]
    while a is not None:
        vp = a.world_state.viewpointId
        if vp in path_to_b_by_viewpoint:
            path_to_b = cons_to_list(path_to_b_by_viewpoint[vp])
            assert path_from_a[-1].world_state.viewpointId == path_to_b[0].world_state.viewpointId
            return path_from_a + path_to_b[1:]
        a = a.prev_inference_state
        path_from_a.append(a)
    raise AssertionError("no common ancestor found")
