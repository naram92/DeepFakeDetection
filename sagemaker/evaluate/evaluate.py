import pandas as pd
import pickle
import argparse
import json
import numpy as np
import logging 
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

from efficientnet_pytorch import EfficientNet
from s3dataset import *

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import LRScheduler, ModelCheckpoint, global_step_from_engine, Checkpoint, DiskSaver, EarlyStopping
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

IMG_SIZE = 224

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="BS",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--num-workers",
        type=float,
        default=2,
        metavar="NW",
        help="number of workers (default: 2)",
    )
    parser.add_argument("--seed", type=int, default=2, metavar="S", help="random seed (default: 2)")

    return parser.parse_args()

def get_data(train_dir, val_dir, test_dir):
    with open(train_dir, "rb") as f:
        df_train = pickle.loads(f.read())
    with open(val_dir, "rb") as f:
        df_val = pickle.loads(f.read())
    with open(test_dir, "rb") as f:
        df_test = pickle.loads(f.read())
    
    return df_train, df_val, df_test

def train(args):
    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda > 0 else "cpu")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        
    val_tranform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std)
    ])
     
    # get data
    _, _, df_test = get_data(args.train, args.validation, args.test)
    # test
    test_dataset = FFPPDataset(df_test, transform=val_tranform)
    test_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # model
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="..")
        
    logger.debug("Loading neumf model.")
    model = torch.jit.load('model.pth')
    model.to(device)
    
    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            y = y.to(device)
            x = x.to(device)
            y_pred = model(x)
            return y_pred, y
    
    evaluator = Engine(eval_step)
    
    # Accuracy and loss metrics are defined
    metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(criterion)
    }
    
    # Attach metrics to the evaluator
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    
    evaluator.add_event_handler(Events.COMPLETED, lambda _: logger.debug('Validation {}'.format(
                        evaluator.state.metrics)))
    evaluator.run(test_loader)
    
    metrics = evaluator.state.metrics
    logger.debug("Accuracy: {}".format(metrics['accuracy']))

    report_dict = {
        "deepfake_detection_metrics": {
            "accuracy": {"value": metrics['accuracy'], "standard_deviation": "NaN"},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

    return


if __name__ == "__main__":
    args = parse_args()
    train(args)
