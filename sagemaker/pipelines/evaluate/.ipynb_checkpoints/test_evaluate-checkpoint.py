import os
import json
import os
import sys
import shutil
import logging
import pathlib
import pickle
import tarfile
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import boto3
import botocore

from ignite.engine import Engine, Events
from evaluate import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def fetch_model(model_data):
    """Untar the model.tar.gz object either from local file system
    or a S3 location

    Args:
        model_data (str): either a path to local file system starts with
        file:/// that points to the `model.tar.gz` file or an S3 link
        starts with s3:// that points to the `model.tar.gz` file

    Returns:
        model_dir (str): the directory that contains the uncompress model
        checkpoint files
    """

    model_dir = "./tmp/model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if model_data.startswith("file"):
        _check_model(model_data)
        shutil.copy2(
            os.path.join(model_dir, "model.tar.gz"), os.path.join(model_dir, "model.tar.gz")
        )
    elif model_data.startswith("s3"):
        # get bucket name and object key
        bucket_name = model_data.split("/")[2]
        key = "/".join(model_data.split("/")[3:])

        s3 = boto3.resource("s3")
        try:
            s3.Bucket(bucket_name).download_file(key, os.path.join(model_dir, "model.tar.gz"))
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print("the object does not exist.")
            else:
                raise

    # untar the model
    tar = tarfile.open(os.path.join(model_dir, "model.tar.gz"))
    tar.extractall(model_dir)
    tar.close()

    return model_dir


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # data augmentation
    pre_trained_mean, pre_trained_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
    val_tranform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std)
    ])
     
    # get data
    train_path = './tmp/data/train.pkl'
    val_path = './tmp/data/val.pkl'
    test_path = './tmp/data/test.pkl'
    
    _, _, df_test = get_data(train_path, val_path, test_path)
    # test
    test_dataset = FFPPDataset(df_test, transform=val_tranform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # model
    # model_path = "./tmp/data/model.tar.gz"
    # with tarfile.open(model_path) as tar:
    #     tar.extractall(path="..")
        
    logger.debug("Loading neumf model.")
    model = torch.jit.load('./tmp/model/model.pth')
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

    output_dir = "./tmp/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
