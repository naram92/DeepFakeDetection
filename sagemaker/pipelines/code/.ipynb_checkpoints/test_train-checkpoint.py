import json
import os
import sys

import boto3
from train import parse_args, train

class Env:
    def __init__(self):
        # simulate container env
        os.environ["SM_MODEL_DIR"] = "./tmp/model"
        os.environ["SM_CHANNEL_TRAIN"] = "./tmp/data/train.pkl"
        os.environ["SM_CHANNEL_VALIDATION"] = "./tmp/data/val.pkl"
        os.environ["SM_CHANNEL_TEST"] = "./tmp/data/test.pkl"
        os.environ["SM_HOSTS"] = '["algo-1"]'
        os.environ["SM_CURRENT_HOST"] = "algo-1"
        os.environ["SM_NUM_GPUS"] = "0"


if __name__ == "__main__":
    Env()
    args = parse_args()
    train(args)
