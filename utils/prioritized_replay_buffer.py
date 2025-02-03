import numpy as np
from gym.spaces import Box, Discrete, Tuple

from replay_buffer import ReplayBuffer

from envs import get_dim


class SegmentTree:
    def __init__(self, max_size, s_dim, a_dim):
        self._s = np.zeros((max_size, s_dim), dtype=np.float32)
        self._s_ = np.zeros((max_size, s_dim), dtype=np.float32)
        self._a = np.zeros((max_size, a_dim), dtype=np.float32)
        self._r = np.zeros((max_size, 1), dtype=np.float32)
        self._done = np.zeros((max_size, 1), dtype=np.uint8)

        self.index = 0
        self.max_size = max_size
        self.full = False
        # TODO tree base
        self.tree_start = 2 ** (max_size - 1).bit_length() - 1
        self.sum_tree = np.zeros((self.tree_start + self.max_size), dtype=np.float32)

        self.max = 1.0

    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        # e.g. [0,1,2,3] -> [1,3,5,7; 2,4,6,8]
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    def _propagate(self, indices):
        parents = (indices - 1) // 2
        # filter out the repeated parents
        unique_parents = np.unique(parents) 
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    # update prioritized value
    def update(self, indices, values):
        self.sum_tree[indices] = values
        self._propagate(indices)
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    # update single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # set new value
        self._propagate_index(index)  # propagate value
        self.max = max(value, self.max)

    def append(self, s, a, r, done, s_, value, **kwargs):
        self._s[self.index] = s
        self._a[self.index] = a
        self._r[self.index] = r
        self._done[self.index] = done
        self._s_[self.index] = s_

        self._update_index(self.index + self.tree_start, value)
        self.index = (self.index + 1) % self.max_size
        self.full = self.full or self.index == 0
        self.max = max(value, self.max)

    def _retrieve(self, indices, values):
        children_indices = indices * 2 + np.expand_dims(
            [1, 2], axis=1
        )  # Make matrix of children indices
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
        successor_indices = children_indices[
            successor_choices, np.arange(indices.size)
        ]  # Use classification to index into the indices matrix
        successor_values = (values - successor_choices * left_children_values)  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)

    # find the data_index with certain prioritized value
    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)

    # get a batch of data
    def get(self, data_index):
        batch = dict()
        batch["s"] = self._s[data_index]
        batch["s_"] = self._s_[data_index]
        batch["a"] = self._a[data_index]
        batch["r"] = self._r[data_index]
        batch["done"] = self._done[data_index]
        return batch

    def total(self):
        return self.sum_tree[0]


class PriorityReplayBuffer(ReplayBuffer):
    # def __init__(self, max_replay_buffer_size, env, env_info_sizes=None):
    def __init__(self, env, max_size=..., batch_size=256, device='cuda'):
        super(PriorityReplayBuffer, self).__init__(env, max_size=int(1e6), batch_size=batch_size, device=device)
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.s_space
        self._a_space = env.a_space

        self.buffer = SegmentTree(
            max_size, get_dim(self._ob_space), get_dim(self._a_space)
        )

    def append(self, s, a, s_, r, done, **kwargs):
        if isinstance(self._a_space, Discrete):
            new_action = np.zeros(self._a_dim)
            new_action[a] = 1
        else:
            new_action = a
        self.buffer.append(s, a, s_, r, done, self.buffer.max)

    def _get_transitions(self, idxs):
        transitions = self.buffer.get(data_index=idxs)
        s = transitions[ : , : self.s_dim]
        a = transitions[ : , self.s_dim : self.s_dim + self.a_dim]
        s_ = transitions[ : , self.s_dim + self.a_dim : 2 * self.s_dim + self.a_dim]
        r = transitions[ : , 2 * self.s_dim + self.a_dim]
        done = transitions[ : , 2 * self.s_dim + self.a_dim + 1]

        s = np.array(s).astype(np.float32)
        a = np.array(a).astype(np.float32)
        s_ = np.array(s_).astype(np.float32)

        transitions_dict = {
            "state": s,
            "action": a,
            "state_": s_,
            "reward": r,
            "done": done
        }

        return transitions_dict

    def _get_samples_from_segments(self, batch_size, p_total):
        segment_length = p_total / batch_size
        segment_starts = np.arange(batch_size) * segment_length
        valid = False
        while not valid:
            samples = (
                np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts
            )
            probs, idxs, tree_idxs = self.buffer.find(samples)
            if np.all(probs != 0):
                valid = True
        batch = self._get_transitions(idxs)
        batch["idxs"] = idxs
        batch["tree_idxs"] = tree_idxs

        return batch

    def random_batch(self, batch_size):
        # return tree_idxs s.t. their values can be updated
        p_total = self.buffer.total()
        return self._get_samples_from_segments(batch_size, p_total)

    def update_priorities(self, idxs, priorities):
        self.buffer.update(idxs, priorities)

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return self.buffer.index

    # TODO OrderedDict
    # def get_diagnostics(self):
    #     return OrderedDict([("size", self.buffer.index)])
