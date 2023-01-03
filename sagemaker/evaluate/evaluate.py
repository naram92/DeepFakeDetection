import os
import json
import os
import sys
import subprocess
import shutil
import logging
import pathlib
import pickle
import tarfile
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import math
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

install('pytorch-ignite')
from ignite.engine import Engine, Events
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.exceptions import NotComputableError


################### METRICS ###################
class HitRate(Metric):
    """
    Hit rate at given position k for each user indicates that
    whether the recommended item is included in the top k ranked list.
    """

    def __init__(self, k, output_transform=lambda x: x):
        super().__init__(output_transform=output_transform)
        self._k = k
        self._scores = None
        self._items = None

    @reinit__is_reduced
    def reset(self):
        super().reset()
        self._scores = []
        self._items = []

    @reinit__is_reduced
    def update(self, output):
        items = output[0].tolist()
        pos_scores = output[1].tolist()
        for i, item in enumerate(items):
            scores = [(item, pos_scores[i])]
            scores.extend(list(zip(output[2][i].tolist(), output[3][i].tolist())))
            scores.sort(key=lambda x: x[0], reverse=True)
            scores.sort(key=lambda x: x[1], reverse=True)
            self._items.append(item)
            self._scores.append(scores)

    @staticmethod
    def _verify_hit_top_n(item_id, items, topn):
        try:
            index = next(i for i, c in enumerate(items) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index + 1

    @sync_all_reduce('_test', '_neg')
    def compute(self):
        if len(self._scores) == 0:
            raise NotComputableError('Empty data')

        hit_count = 0
        for i, item_id in enumerate(self._items):
            items = [score[0] for score in self._scores[i]]
            hit, _ = self._verify_hit_top_n(item_id, items, self._k)
            hit_count += hit

        return hit_count / len(self._items)


class NDCG(HitRate):
    """
    Normalized Discounted Cumulative Gain
    """

    def __init__(self, k, output_transform=lambda x: x):
        super().__init__(k, output_transform)

    @sync_all_reduce('_test', '_neg')
    def compute(self):
        if len(self._scores) == 0:
            raise NotComputableError('Empty data')

        ndcg_count = 0
        for i, item_id in enumerate(self._items):
            items = [score[0] for score in self._scores[i]]
            hit, rank = self._verify_hit_top_n(item_id, items, self._k)
            if hit == 1:
                ndcg_count += math.log(2) / math.log(1 + rank)

        return ndcg_count / len(self._items)

################### Sequence Dataset ###################
class SequenceDataset(Dataset):
    """
    Sequential Dataset with Negative Sampling
    
    Args:
        users (list): list of users
        items (list): list of items
        l (int): previous item length
        num_users (int): number of users
        num_items (int): number of items
        candidates (dict): candidate set of each user  
    
    Returns :
        In each sample, it outputs the user identity, his previous  𝐿  interacted items as a sequence and the next item he interacts as the target. 
    """
    
    def __init__(self, users, items, l, num_users, num_items, candidates):
        user_ids, item_ids = torch.tensor(users), torch.tensor(items)
        sort_idx = sorted(range(len(user_ids)), key=lambda k: user_ids[k])
        u_ids, i_ids = user_ids[sort_idx], item_ids[sort_idx]
        temp, self.cand = {}, candidates
        self.all_items = set([i for i in range(num_items)])
        [temp.setdefault(u_ids[i].item(), []).append(i) for i, _ in enumerate(u_ids)]
        temp = sorted(temp.items(), key=lambda x: x[0])
        u_ids = torch.tensor([i[0] for i in temp])
        idx = torch.tensor([i[1][0] for i in temp])
        self.ns = ns = sum([c - l if c >= l + 1 else 1 for c in [len(i[1]) for i in temp]])

        self.seq_items = torch.zeros(ns, l).long()
        self.seq_users = torch.zeros(ns).long()
        self.seq_tgt = torch.zeros(ns).long()
        self.test_seq = torch.zeros(ns, l).long()
        test_users, _uid = torch.zeros(num_users, l).long(), None
        for i, (uid, i_seq) in enumerate(self._seq(u_ids, i_ids, idx, l + 1)):
            if uid != _uid:
                self.test_seq[uid][:] = i_seq[-l:]
                test_users[uid], _uid = uid, uid
            self.seq_tgt[i] = i_seq[-1]
            self.seq_items[i][:], self.seq_users[i] = i_seq[:l], uid

    @staticmethod
    def _win(tensor, window_size, step_size=1):
        if len(tensor) - window_size >= 0:
            for i in range(len(tensor), 0, -step_size):
                if i - window_size >= 0:
                    yield tensor[i - window_size: i]
                else:
                    break
        else:
            yield tensor

    def _seq(self, u_ids, i_ids, idx, max_len):
        for i in range(len(idx)):
            stop_idx = None if i >= len(idx) - 1 else int(idx[i + 1])
            for s in self._win(i_ids[int(idx[i]):stop_idx], max_len):
                yield (int(u_ids[i]), s)

    def __len__(self):
        return self.ns

    def __getitem__(self, idx):
        neg = list(self.all_items - set(self.cand[self.seq_users[idx].item()]))
        i = random.randint(0, len(neg) - 1)
        return self.seq_users[idx], self.seq_items[idx], self.seq_tgt[idx], neg[i]


class SequenceTestDataset(Dataset):
    def __init__(self, users, items, num_items, candidates, seq_iter):
        self.users = users
        self.items = items
        self.candidates = candidates
        self.seq_iter = seq_iter
        self.all = set([i for i in range(num_items)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        neg_items = list(self.all - set(self.candidates[int(self.users[idx])]))
        np.random.shuffle(neg_items)
        neg_items = neg_items[:10]

        return torch.tensor(self.users[idx]), self.seq_iter[self.users[idx], :].clone().detach(), torch.tensor(self.items[idx]), torch.tensor(neg_items)


def get_engine(model, hit_max):
    def evaluate_step(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(tensor.to(device) for tensor in batch)
            score_pos = model.forward(*batch[:-1])
            scores_neg = torch.zeros(batch[-1].size(1), batch[0].size(0)).float()
            for i in range(batch[-1].size(1)):
                score_neg = model.forward(*batch[:-2], batch[-1][:, i])
                scores_neg[i] = score_neg.view(-1)

            return batch[-2], score_pos.view(-1), batch[-1], scores_neg.transpose(0, 1)
    
    evaluator = Engine(evaluate_step)

    metrics = dict((f'hit@{i + 1}', HitRate(i + 1)) for i in range(hit_max))
    metrics.update(dict((f'ndcg@{i + 1}', NDCG(i + 1)) for i in range(hit_max)))

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator

def get_data(train_dir, test_dir):
    # training dataset
    train_table = pq.read_table(train_dir)
    train_interactions = train_table.to_pandas()
    # testing dataset
    test_table = pq.read_table(test_dir)
    test_interactions = test_table.to_pandas()
    # training + testing dataset
    interactions = pd.concat([train_interactions, test_interactions], ignore_index=True)
    num_users = interactions.user_id.unique().shape[0]
    num_items = interactions.item_id.unique().shape[0]
    
    return num_users, num_items, train_interactions, test_interactions

def load_data(data):
    users, items = [], []
    inter = {}
    for line in data.itertuples():
        user_index, item_index = int(line[1]), int(line[2])
        users.append(user_index)
        items.append(item_index)
        inter.setdefault(user_index, []).append(item_index)
    return users, items, inter


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    
    #args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prev-item-len",
        type=int,
        default=3,
        metavar="PIL",
        help="pevious item length (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="BS",
        help="input batch size for testing (default: 16)",
    )
    args = parser.parse_args()
    
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="..")
        
    logger.debug("Loading neumf model.")
    model = torch.jit.load('model.pth')
    model.to(device)

    logger.debug("Loading test input data.")
    test_path = "/opt/ml/processing/test/"
    train_path = "/opt/ml/processing/train/"
    num_users, num_items, train_interactions, test_interactions = get_data(train_path, test_path)
    
    users_train, items_train, candidates = load_data(train_interactions)
    users_test, items_test, test_candidates = load_data(test_interactions)
    for u in test_candidates:
        test_candidates[u].extend(candidates[u])
    
    train_dataset = SequenceDataset(users_train, items_train, args.prev_item_len, num_users, num_items, candidates)
    # Test loader
    test_loader = DataLoader(SequenceTestDataset(users_test, items_test, num_items, test_candidates, train_dataset.test_seq), args.batch_size, shuffle=False, num_workers=0)

    evaluator = get_engine(model, 10)
    evaluator.add_event_handler(Events.COMPLETED, lambda _: logger.debug('Validation {}'.format(
                        evaluator.state.metrics)))
    evaluator.run(test_loader)
    
    metrics = evaluator.state.metrics
    logger.debug("HIT@5: {}".format(metrics['hit@5']))
    logger.debug("NGC@5: {}".format(metrics['ndcg@5']))

    report_dict = {
        "recommender_system_metrics": {
            "ndcg@5": {"value": metrics['ndcg@5'], "standard_deviation": "NaN"},
            "hit@5": {"value": metrics['hit@5'], "standard_deviation": "NaN"},
            "ndcg@10": {"value": metrics['ndcg@10'], "standard_deviation": "NaN"},
            "hit@10": {"value": metrics['hit@10'], "standard_deviation": "NaN"},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
