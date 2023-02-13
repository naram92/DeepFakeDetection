import logging
import os
import argparse
from datetime import datetime
import copy
import pandas as pd
import numpy as np
from face_extract import preprocess_ffpp, extract_faces

FFPP_SRC = 'dev_datasets/'
# FFPP_SRC = 'datasets/'
VIDEODF_SRC = os.path.join(FFPP_SRC, 'ffpp_videos.pkl')

BLAZEFACE_WEIGHTS = os.path.join(FFPP_SRC, 'blazeface/blazeface.pth')
BLAZEFACE_ANCHORS = os.path.join(FFPP_SRC, 'blazeface/anchors.npy') 

FACES_DST = os.path.join(FFPP_SRC, 'extract_faces')
FACESDF_DST = os.path.join(FACES_DST, 'ffpp_faces.pkl')
CHECKPOINT_DST = os.path.join(FACES_DST, 'checkpoint')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def make_splits(df_faces, train_ratio=0.7, val_ratio=0.15):
    """
    Splitting the dataset into three subsets: train, validation, and test
    """
    random_original_videos = np.random.permutation(df_faces[(df_faces['label'] == 0)]['video'].unique())
    train_num = int(len(random_original_videos) * train_ratio)
    val_num = int(len(random_original_videos) * val_ratio)
    train_original = random_original_videos[:train_num]
    val_original = random_original_videos[train_num: train_num + val_num]
    test_original = random_original_videos[train_num + val_num:]
    
    df_train = pd.concat([df_faces[df_faces['original'].isin(train_original)], 
                          df_faces[df_faces['video'].isin(train_original)]], ignore_index=True)
    df_val = pd.concat([df_faces[df_faces['original'].isin(val_original)], 
                        df_faces[df_faces['video'].isin(val_original)]], ignore_index=True)
    df_test = pd.concat([df_faces[df_faces['original'].isin(test_original)], 
                         df_faces[df_faces['video'].isin(test_original)]], ignore_index=True)
    
    return df_train, df_val, df_test

def main():
    logger.info("Starting preprocessing.")
    
    #args
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_per_video', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--face_size', type=int)
    args = parser.parse_args()
    
    # preprocess ff++ data
    preprocess_ffpp(FFPP_SRC, VIDEODF_SRC)
    # Run extraction
    df_faces = extract_faces(FFPP_SRC, VIDEODF_SRC, FACES_DST, FACESDF_DST,  CHECKPOINT_DST, BLAZEFACE_WEIGHTS, BLAZEFACE_ANCHORS, 
                             args.frames_per_video, args.batch_size, args.face_size)
    # Split data
    df_train, df_val, df_test = make_splits(df_faces)
    print("Saving train, val and test dataframes")
    df_train.to_pickle("/opt/ml/processing/output/train/train.pkl")
    df_val.to_pickle("/opt/ml/processing/output/validation/val.pkl")
    df_test.to_pickle("/opt/ml/processing/output/test/test.pkl")
    

if __name__ == '__main__':
    main()