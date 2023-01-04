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

# def install(package):
#     subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# install('efficientnet-pytorch')
# install('pytorch-ignite')
from efficientnet_pytorch import EfficientNet

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import LRScheduler, ModelCheckpoint, global_step_from_engine, Checkpoint, DiskSaver, EarlyStopping
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear

import boto3
from PIL import Image
import os
from pathlib import Path 
import numpy as np

FFPP_SRC = 'dev_datasets/'
FACES_DST = os.path.join(FFPP_SRC, 'extract_faces')

s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')
bucket_name = 'deepfake-detection'
bucket = s3_resource.Bucket(bucket_name)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

IMG_SIZE = 224

class FFPPDataset(Dataset):
    def __init__(self, df_faces, faces_dir=FACES_DST, transform=None):
        super().__init__()
        self.faces_dir = Path(faces_dir)
        self.data, self.targets = df_faces['path'], df_faces['label']
        self.transform = transform
        
    def __getitem__(self, index):
        img_path, target = self.data[index], self.targets[index]
        target = np.array([target,]).astype(np.float32)
        
        file_stream = bucket.Object(str(self.faces_dir.joinpath(img_path))).get()['Body']
        img = Image.open(file_stream)
        
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.data)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # data augmentation
    pre_trained_mean, pre_trained_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
    val_tranform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std)
    ])
     
    # get data
    train_path = '/opt/ml/processing/train/train.pkl'
    val_path = '/opt/ml/processing/validation/val.pkl'
    test_path = '/opt/ml/processing/train/test.pkl'
    
    _, _, df_test = get_data(train_path, val_path, test_path)
    # test
    test_dataset = FFPPDataset(df_test, transform=val_tranform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # model
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="..")
        
    logger.debug("Loading model")
    model = torch.jit.load('model.pth')
    model.to(device)
    
    # loss_function
    criterion = nn.BCEWithLogitsLoss()
    
    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            y = y.to(device)
            x = x.to(device)
            y_pred = model(x)
            return y_pred, y
    
    evaluator = Engine(eval_step)
    
    def thresholded_output_transform(output):
        y_pred, y = output
        y_pred = torch.sigmoid(y_pred)
        y_pred = torch.round(y_pred)
        return y_pred, y
    
    # Accuracy and loss metrics are defined
    metrics = {
        "accuracy": Accuracy(output_transform=thresholded_output_transform),
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
