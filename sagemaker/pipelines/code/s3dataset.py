# from torch.utils.data import IterableDataset
# from torchdata.datapipes.iter import IterableWrapper, S3FileLoader
from torch.utils.data import Dataset
import boto3
from PIL import Image
import os
from pathlib import Path 
import numpy as np

FFPP_SRC = 'dev_datasets/'
# FFPP_SRC = 'datasets/'
FACES_DST = os.path.join(FFPP_SRC, 'extract_faces')

s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')
bucket_name = 'deepfake-detection'
bucket = s3_resource.Bucket(bucket_name)

# class S3FFPPDataset(IterableDataset):
#     def __init__(self, df, faces_dir=FACES_DST, shuffle_urls=False, transform=None):
#         super().__init__()
#         self.data = df
#         self.paths = IterableWrapper(df['s3_path'].unique().tolist()).list_files_by_s3()
#         self.targets = IterableWrapper(df['label'])
#         if shuffle_urls:
#             self.sharded_s3_urls = self.paths.shuffle().sharding_filter()
#             self.s3_files = S3FileLoader(self.sharded_s3_urls)
#         else:
#             self.s3_files = S3FileLoader(self.paths)
#         self.transform = transform
    
#     def data_generator(self):
#         try:
#             while True:
#                 url, stream = next(self.s3_files_iterator)
#                 target = next(self.targets_iterator)
                
#                 target = np.array([target,]).astype(np.float32)
#                 img = Image.open(stream)
                
#                 if self.transform is not None:
#                     img = self.transform(img)
#                 yield img, target

#         except StopIteration:
#             return

#     def __iter__(self):
#         self.s3_files_iterator = iter(self.s3_files)
#         self.targets_iterator = iter(self.targets)
#         return self.data_generator()

#     def __len__(self):
#         return len(self.data)
    
    
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