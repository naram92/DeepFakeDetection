import boto3
import botocore

import io
import os
import argparse
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path, PurePosixPath
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from blazeface import BlazeFace
import cv2
from PIL import Image

s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')
bucket_name = 'deepfake-detection'
bucket = s3_resource.Bucket(bucket_name)

def preprocess_ffpp(source_dir, video_dataset_path):
    """
    Preprocessing video dataset : Set the label of each video {0 for real video, 
    1 for fake video} and the video original of fake videos.
    :param source_dir: the parent directory that contains all videos (real or 
                        fake)
    :param video_dataset_path: Path to save the videos DataFrame[path, name, 
                                label, original]
    """ 
    try:
        s3_resource.Object(bucket_name, video_dataset_path).load()
        file_exists = True
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            file_exists = False
        else:
            raise

    # Charger le fichier de checkpoint depuis S3 s'il existe
    if file_exists:
        df_videos = pickle.loads(s3_resource.Bucket(bucket_name).Object(video_dataset_path).get()['Body'].read())
    else :
        # Si le fichier n'existe pas dans S3, vous pouvez créer le DataFrame de la façon suivante :
        print('Creating video DataFrame')
        # df_videos = pd.DataFrame(columns=['path', 'label', 'name', 'original'])

        # Initialisez une liste vide pour stocker les chemins des fichiers .mp4
        mp4_files = []

        # Appelez la méthode list_objects_v2 de l'objet client S3 en spécifiant le paramètre ContinuationToken
        # lors de chaque itération, jusqu'à ce qu'il n'y ait plus d'objets à récupérer 
        objects_list = s3_client.list_objects_v2(
            Bucket=bucket_name, Prefix=source_dir
        )
        while True:
            # Parcourez la liste des objets et ajoutez le chemin de chaque fichier mp4 à la liste
            for obj in objects_list['Contents']:
                # Vérifiez si l'objet est un fichier mp4
                if obj['Key'].endswith('.mp4'):
                    # Ajoutez le chemin du fichier à la liste
                    mp4_files.append(obj['Key'])
            
            # Vérifiez si il y a une suite de résultats
            if 'NextContinuationToken' in objects_list:
                # Si oui, récupérez la suite des résultats en spécifiant le ContinuationToken
                objects_list = s3_client.list_objects_v2(
                    Bucket=bucket_name, Prefix=source_dir, ContinuationToken=objects_list['NextContinuationToken']
                )
            else:
                # Si non, sortez de la boucle
                break
        
        # Créez le DataFrame en utilisant la liste des chemins de fichiers .mp4
        df_videos = pd.DataFrame({'path': mp4_files})
        # Enlevez le repertoire racine datasets/ dans le path
        df_videos['path'] = df_videos['path'].replace('datasets/', '', regex=True)
        # Convertissez les chaînes de caractères en objets PurePosixPath
        df_videos['path'] = df_videos['path'].apply(lambda x: PurePosixPath(x))           

        # 1 if fake, otherwise 0
        df_videos['label'] = df_videos['path'].map(
            lambda x: 1 if x.parts[0] == 'manipulated_sequences' else 0)
        
        source = df_videos['path'].map(lambda x: x.parts[1]).astype('category')
        df_videos['name'] = df_videos['path'].map(lambda x: x.with_suffix('').parts[-1])
        df_videos['path'] = df_videos['path'].map(lambda x: str(x))

        df_videos['original'] = -1 * np.ones(len(df_videos), dtype=np.int16)
        # Mettre dans la colonne original l'index de l'original des fakes
        df_videos.loc[(df_videos['label'] == 1) & (source != 'DeepFakeDetection'), 'original'] = \
            df_videos[(df_videos['label'] == 1) & (source != 'DeepFakeDetection')]['name'].map(
                lambda x: df_videos.index[np.flatnonzero(df_videos['name'] == x.split('_')[0])[0]]
            )
        df_videos.loc[(df_videos['label'] == 1) & (source == 'DeepFakeDetection'), 'original'] = \
            df_videos[(df_videos['label'] == 1) & (source == 'DeepFakeDetection')]['name'].map(
                lambda x: df_videos.index[
                    np.flatnonzero(df_videos['name'] == x.split('_')[0] + '__' + x.split('__')[1])[0]]
            )
    
        # Enregistrez le DataFrame dans S3
        print('Saving video DataFrame to {}'.format(video_dataset_path))
        buf = io.BytesIO()
        df_videos.to_pickle(buf)
        buf.seek(0)
        bucket.upload_fileobj(buf, video_dataset_path)
    
    print('Real videos: {:d}'.format(sum(df_videos['label'] == 0)))
    print('Fake videos: {:d}'.format(sum(df_videos['label'] == 1)))
    
def extract_faces_on_video(video_df, source_dir, faces_dir, checkpoint_dir, 
                           blazeface, num_frames, face_size=224, margin=0.25):
    """
    This function extracts `num_frames` frames in the videos that contain a face.
    :param video_df: the DataFrame that contains all informations about the 
                    datasets. It has the following columns: [path, name, 
                    label, original].
    :param source_dir: the parent directory that contains the datasets
    :param faces_dir: the directory path to save the extracted faces from the 
                    datasets
    :param checkpoint_dir: the directory path to save the DataFrame[path, label,
                    video, original, frame_index, score, detection] of the 
                    extracted faces
    :param blazeface: a Balazeface object that will be used as face detector in
                    all frames
    :param num_frames: number of frames to extract in each video.
    :param face_size (default = 224) : each frame extracted will have the size
                    face_size x face_size
    :param margin (default = 0.25) : Offset margin of face detection.
    """
    video_idx, video_df = video_df
    faces_checkpoint_path = Path(checkpoint_dir).joinpath(video_df['path'].split('.')[0] + '_faces.pkl')
    
    try:
        s3_resource.Object(bucket_name, str(faces_checkpoint_path)).load()
        file_exists = True
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            file_exists = False
        else:
            raise
        
    if file_exists: 
        faces = pickle.loads(s3_resource.Bucket(bucket_name).Object(str(faces_checkpoint_path)).get()['Body'].read())
        return faces
        
    else :
        # Télécharger la vidéo depuis S3 dans un buffer
        video_path = Path(source_dir).joinpath(video_df['path'])
        url = s3_client.generate_presigned_url(ClientMethod='get_object', Params={ 'Bucket': bucket_name, 'Key': str(video_path) })
        reader =  cv2.VideoCapture(url)
        # Obtenir le nombre de frames de la vidéo
        frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        # Obtenir un tableau d'indices de frames uniformément répartis dans la vidéo
        frame_idx = np.unique(np.linspace(0, frame_count - 1, num_frames, dtype=int))
        # Get the frames choosen
        frames, idx = [], 0
        # Tant que la vidéo peut être lue
        while reader.grab():
            if idx in frame_idx:
                ret, frame = reader.retrieve()
                if not ret or frame is None:
                    print("Error retrieving frame %d from movie %s" % (frame_idx, path))
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        frames = np.stack(frames) # Empiler les frames dans un tableau NumPy
        
        # Obtenir la largeur et la hauteur cibles pour le modèle BlazeFace
        target_w, target_h = blazeface.input_size
        # Pour chaque frame, obtenir plusieurs tuiles de taille target_w x target_h
        num_frames, height, width, _ = frames.shape
            # Définir la taille de chaque tuile en prenant la plus petite valeur parmi height, width et 720
        split_size = min(height, width, 720)
        x_step = (width - split_size) // 2
        y_step = (height - split_size) // 2
        num_h = (height - split_size) // y_step + 1 if y_step > 0 else 1
        num_w = (width - split_size) // x_step + 1 if x_step > 0 else 1

        tiles = np.zeros((num_frames * num_h * num_w, target_h, target_w, 3), 
                        dtype=np.uint8)
        i = 0
        for f in range(num_frames):
            y = 0
            for _ in range(num_h):
                x = 0
                for __ in range(num_w):
                    # Découper une tuile à partir de la frame actuelle
                    crop = frames[f, y:y + split_size, x:x + split_size, :]
                    # Redimensionner la tuile à la taille cible en utilisant une interpolation par aire
                    tiles[i] = cv2.resize(crop, (target_w, target_h), 
                                        interpolation=cv2.INTER_AREA)
                    x += x_step
                    i += 1
                y += y_step

        # Run the face detector. The result is a list of PyTorch tensors
        detections = blazeface.predict_on_batch(tiles, apply_nms=False)
        # Convert the detections from 128x128 back to the original frame size
        for i in range(len(detections)):
            # ymin, xmin, ymax, xmax
            for k in range(2):
                detections[i][:, k * 2] = (detections[i][:, k * 2] * target_h) * split_size / target_h
                detections[i][:, k * 2 + 1] = (detections[i][:, k * 2 + 1] * target_w) * split_size / target_w

        # Because we have several tiles for each frame, combine the predictions from these tiles.
        combined_detections = []
        i = 0
        for f in range(num_frames):
            detections_for_frame = []
            y = 0
            for _ in range(num_h):
                x = 0
                for __ in range(num_w):
                    # Adjust the coordinates based on the split positions.
                    if detections[i].shape[0] > 0:
                        for k in range(2):
                            detections[i][:, k * 2] += y
                            detections[i][:, k * 2 + 1] += x
                    
                    detections_for_frame.append(detections[i])
                    x += x_step
                    i += 1
                y += y_step
                
            combined_detections.append(torch.cat(detections_for_frame))
        if len(combined_detections) == 0:
            return None
        detections = blazeface.nms(combined_detections)
        # Crop the faces out of the original frame.
        faces = []
        for i in range(len(detections)):
            offset = torch.round(margin * (detections[i][:, 2] - detections[i][:, 0])) # margin 0.2
            detections[i][:, 0] = torch.clamp(detections[i][:, 0] - offset * 2, min=0)  # ymin
            detections[i][:, 1] = torch.clamp(detections[i][:, 1] - offset, min=0)  # xmin
            detections[i][:, 2] = torch.clamp(detections[i][:, 2] + offset, max=height)  # ymax
            detections[i][:, 3] = torch.clamp(detections[i][:, 3] + offset, max=width)  # xmax
            
            # Get the first best scored face
            score, face, detection = 0, None, None
            for j in range(len(detections[i])):
                if score < detections[i][j][16].cpu():
                    detection = detections[i][j].cpu()
                    ymin, xmin, ymax, xmax = detection[:4].cpu().numpy().astype(int)
                    face = frames[i][ymin:ymax, xmin:xmax, :]
                    score = detection[16]
                    break
            if face is not None:
                image = Image.fromarray(face)
                # Crop the image to face_size x face_size
                top, left, bottom, right = detection[:4].cpu().numpy().astype(int)
                x_ctr = (left + right) // 2
                y_ctr = (top + bottom) // 2
                new_top = max(y_ctr - face_size // 2, 0)
                new_bottom = min(new_top + face_size, height)
                new_left = max(x_ctr - face_size // 2, 0)
                new_right = min(new_left + face_size, width)
                image.crop([new_left, new_top, new_right, new_bottom])
                # Save image
                face_path = Path(faces_dir).joinpath(video_df['path']).joinpath('frame_{}.jpg'.format(frame_idx[i]))
                buf = io.BytesIO()
                image.save(buf, format="jpeg")
                object = bucket.Object(str(face_path))
                object.put(Body=buf.getvalue())
                faces.append({
                    'path': str(Path(video_df['path']).joinpath('frame_{}.jpg'.format(frame_idx[i]))),
                    'label': video_df['label'],
                    'video': video_idx,
                    'original': video_df['original'],
                    'frame_index': frame_idx[i],
                    'score': float(score.numpy()),
                    'detection': detection[:4].cpu().numpy().astype(int)
                })
            # Save checkpoint
            buf = io.BytesIO()
            pd.DataFrame(faces).to_pickle(buf)
            buf.seek(0)
            bucket.upload_fileobj(buf, str(faces_checkpoint_path))

        return faces

def extract_faces(source_dir, videos_df, faces_dir, faces_df, checkpoint_dir, 
                  frames_per_video=15, batch_size=32, face_size=224, thread_num=7):
    """
    This function extracts all frames in the dataset that contain a face.
    :param source_dir: the parent directory that contains the datasets
    :param videos_df: the path of the DataFrame containing all informations 
                    about the videos in the dataset.
    :param faces_dir: the directory path to save the extracted faces from the 
                    datasets
    :param faces_df: the path to save the DataFrame containing all informations 
                    about the extracted faces.
    :param checkpoint_dir: the directory path to save the DataFrame[path, label,
                    video, original, frame_index, score, detection] of the 
                    extracted faces
    :param frames_per_video (default = 15): number of frames to extract in each
                    video.
    :param batch_size (default = 16): batch size of videos to treat together.
    :param face_size (default = 224) : each frame extracted will have the size
                    face_size x face_size
    :thread_num (default = 4): number of threads to be used during the 
                    extraction.
    """
    # On vérifie si ffpp_faces.pkl existe
    try:
        s3_resource.Object(bucket_name, faces_df).load()
        file_exists = True
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            file_exists = False
        else:
            raise

    # Charger le fichier de checkpoint depuis S3 s'il existe
    if file_exists:
        df_faces = pickle.loads(s3_resource.Bucket(bucket_name).Object(faces_df).get()['Body'].read())
        print('We got {} faces'.format(len(df_faces)))
        print('Faces DataFrame Loaded')
        return df_faces
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print('Loading video DataFrame')
    df_videos = pickle.loads(s3_resource.Bucket(bucket_name).Object(videos_df).get()['Body'].read())
    
    print('Loading Blazeface model')
    blazeface_net = BlazeFace().to(device)
    blazeface_net.load_weights(io.BytesIO(s3_client.get_object(Bucket=bucket_name, Key=BLAZEFACE_WEIGHTS)['Body'].read()))
    blazeface_net.load_anchors(io.BytesIO(s3_client.get_object(Bucket=bucket_name, Key=BLAZEFACE_ANCHORS)['Body'].read()))    
    blazeface_net.min_score_thresh = 0.8
    
    ## Face extraction
    with ThreadPoolExecutor(thread_num) as pool:
        for batch_idx0 in tqdm(np.arange(start=0, stop=len(df_videos), step=batch_size),
                               desc='Extracting faces'):
            list(pool.map(partial(extract_faces_on_video,
                          source_dir=source_dir,
                          faces_dir=faces_dir,
                          checkpoint_dir=checkpoint_dir,
                          blazeface=blazeface_net,
                          num_frames=frames_per_video,
                          face_size=face_size,
                          ),
                          df_videos.iloc[batch_idx0:batch_idx0 + batch_size].iterrows()))
    
    faces_dataset = []
    for _, df in tqdm(df_videos.iterrows(), total=len(df_videos), desc='Collecting faces results'):
        face_checkpoint = Path(checkpoint_dir).joinpath(df['path'].split('.')[0] + '_faces.pkl')
        try:
            s3_resource.Object(bucket_name, str(face_checkpoint)).load()
            face_checkpoint_exists = True
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                file_exists = False
            else:
                raise
        if face_checkpoint_exists:
            df_face = pickle.loads(s3_resource.Bucket(bucket_name).Object(str(face_checkpoint)).get()['Body'].read())
            faces_dataset.append(df_face)
        else:
            print(f'Checkpoint file {face_checkpoint} does not exist')
            
    df_faces = pd.concat(faces_dataset, axis=0)
    buf = io.BytesIO()
    df_faces.to_pickle(buf)
    buf.seek(0)
    bucket.upload_fileobj(buf, faces_df)
    print('We got {} faces'.format(len(df_faces)))
    print('Completed!')
    return df_faces