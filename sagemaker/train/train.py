import pandas as pd
import pickle
import argparse
import json
import numpy as np

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
import ignite.contrib.engines.common as common

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
        "--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 2)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--num-workers",
        type=float,
        default=2,
        metavar="NW",
        help="number of workers (default: 2)",
    )
    parser.add_argument("--seed", type=int, default=2, metavar="S", help="random seed (default: 2)")

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

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
        
    # data augmentation
    pre_trained_mean, pre_trained_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
        transforms.RandomPerspective(distortion_scale=0.2),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(scale=(0.02, 0.16), ratio=(0.3, 1.6)),
        transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std)
    ])

    val_tranform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=pre_trained_mean, std=pre_trained_std)
    ])

    # get data
    df_train, df_val, _ = get_data(args.train, args.validation, args.test)
    # train 
    train_dataset = FFPPDataset(df_train, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=args.num_workers)
    # validation
    valid_dataset = FFPPDataset(df_val, transform=val_tranform)
    val_loader = DataLoader(valid_dataset, batch_size=32, num_workers=args.num_workers)
    
    # model
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = nn.Linear(model._conv_head.out_channels, 1)

    for name, param in model.named_parameters():
        if not name.startswith('_fc'):
            param.requires_grad = False
    
    # loss_function
    criterion = nn.BCEWithLogitsLoss()
    # optimizer
    params = (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.learning_rate), 
                                                  (args.epochs * len(train_loader), 0.0)])

    # Setup pytorch-ignite trainer engine
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            y = y.to(device)
            x = x.to(device)
            y_pred = model(x)
            return y_pred, y
    
    trainer = Engine(train_step)
    evaluator = Engine(eval_step)
    
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    
    # Accuracy and loss metrics are defined
    metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(criterion)
    }
    
    # Attach metrics to the evaluator
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    
    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(trainer, ['loss'])
    
    # Early stoping 
    # if the loss of the validation set does not decrease in 5 epochs, the training process will stop early
    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    handler = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        pbar.log_message(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_loss))
    
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    
    def score_function(engine):
        return engine.state.metrics["accuracy"]
   
    to_save = {'model': model.cpu()}

    # Save the best model based on accuracy metric
    handler = Checkpoint(
        to_save, DiskSaver(args.model_dir, create_dir=False),
        n_saved=1, filename_prefix='model',
        score_function=score_function, score_name="accuracy",
        global_step_transform=global_step_from_engine(trainer),
        filename_pattern='{filename_prefix}.pth'
    )
    
    evaluator.add_event_handler(Events.COMPLETED, handler)
    
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = validation_evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        pbar.log_message(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_loss))
        pbar.n = pbar.last_print_n = 0
        
    trainer.add_event_handler(Events.COMPLETED, log_validation_results)
    
    trainer.run(train_loader, max_epochs=args.epochs)
    
    # save the model to torchscript format
    save_model_to_torchscript(model, args.model_dir, train_loader)

    return

def save_model_to_torchscript(model, model_dir, loader):
    # save file path
    save_file_path = os.path.join(model_dir,'model.pth')
    batch = next(iter(loader))
    # load the model 
    model.load_state_dict(torch.load(f=save_file_path))
    # Set the model into eval mode
    model.eval()
    # trace the model
    traced_module = torch.jit.trace(model, batch[:-1])
    # save the model 
    traced_module.save(save_file_path)
    return

if __name__ == "__main__":
    args = parse_args()
    train(args)

