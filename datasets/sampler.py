from collections import defaultdict
import math
import numpy as np
from torch.utils.data import Sampler
from torchvision.datasets import ImageFolder
from itertools import combinations


class EpisodeSampler:
    """
    Stable sampler for support and query indices. Used by episodic batch sampler, so that the support and query sets
    can be sampled from independent data loaders using the same splits, i.e., such that support and query do not overlap.
    """

    def __init__(self, dataset: ImageFolder, n_way: int, n_shot: int, n_query_shot: int, n_episodes: int,
                 seed: int = 0):
        self.dataset = dataset
        self.n_classes = len(dataset.classes)
        self.w = n_way
        self.s = n_shot # 160 or 200
        self.q = n_query_shot # 0
        self.n_episodes = n_episodes # 1
        self.seed = seed
        self.class_task = math.floor(self.n_classes/self.w)
        self.task_random = np.random.randint(self.class_task)

        rs = np.random.RandomState(seed) # random generator 객체 자체를 가지고옴
        self.episode_seeds = []
        for i in range(n_episodes):
            self.episode_seeds.append(rs.randint(2 ** 32 - 1))

        self.indices_by_class = defaultdict(list) # default값이 list형식인 dict
        for index, (path, label) in enumerate(dataset.samples):
            self.indices_by_class[label].append(index) # {label : [index, index], ...} # 같은 label(class)인 경우 리스트 내에 들어감~
        
        #self.class_comb = combinations(range(self.n_classes), self.w)

        # for i in range(self.n_classes):
        #     print("class", i, ":", len(self.indices_by_class[i]))

    def __getitem__(self, index):
        """
        :param index:
        :return: support: ndarray[w, s], query: ndarray[w ,q]
        """
        rs = np.random.RandomState(self.episode_seeds[index])
        # fix class 
        # if self.dataset.name == 'CropDisease':
        #     selected_classes = [3, 15, 16, 24, 28]
        # elif self.dataset.name == 'ISIC':
        #     selected_classes = [0, 1, 2, 3, 4]
        # else:
        #     selected_classes = rs.permutation(self.n_classes)[:self.w]

        selected_classes = rs.permutation(self.n_classes)[:self.w] # fix classes
        # selected_classes = [self.task_random * self.w + i for i in range(self.w)]
        # selected_classes = next(self.class_comb)
        # print(selected_classes)

        indices = []
        # support_indices = []
        # query_indices = []
        
        for cls in selected_classes: # 해당 cls에 해당하는 sample index 중에서 sampling
            # support_indices.append(rs.choice(self.indices_by_class[cls][15:], self.s, replace=False))
            # query_indices.append(rs.choice(self.indices_by_class[cls][:15], self.q, replace=False))
            indices.append(rs.choice(self.indices_by_class[cls], self.s + self.q, replace=False)) # 비복원 추출 # 인덱스를 추출해서 넣어야할듯 
        
        # support = np.stack(support_indices)
        # query = np.stack(query_indices)
        episode = np.stack(indices) # [800, 3, 224, 224]
        # support query split
        support = episode[:, :self.s]
        query = episode[:, self.s:] 
        return support, query

    def __len__(self):
        return self.n_episodes


class EpisodicBatchSampler(Sampler):
    """
    For each epoch, the same batch is yielded repeatedly. For batch-training within episodes, you need to divide up the
    sampled data (from the dataloader) into further smaller batches.

    For classification-based training, note that you need to reset the class indices to [0, 0, ..., 1, ..., w-1]. Note
    that this is why inter-episode batches are not supported by the sampler: it's harder to reset the class indices.
    """

    def __init__(self, dataset: ImageFolder, n_way: int, n_shot: int, n_query_shot: int, n_episodes: int, support: bool,
                 n_epochs=1, seed=0):
        super().__init__(dataset)
        self.dataset = dataset

        self.w = n_way
        self.s = n_shot
        self.q = n_query_shot
        self.episode_sampler = EpisodeSampler(dataset, n_way, n_shot, n_query_shot, n_episodes, seed)

        self.n_episodes = n_episodes
        self.n_epochs = n_epochs
        self.support = support

    def __len__(self):
        return self.n_episodes * self.n_epochs

    def __iter__(self):
        for i in range(self.n_episodes):
            support, query = self.episode_sampler[i]
            indices = support if self.support else query
            indices = indices.flatten()
            for j in range(self.n_epochs):
                yield indices
