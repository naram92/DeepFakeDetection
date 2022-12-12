# DeepFakeDetection
## Dataset
[FaceForensics++](https://github.com/ondyari/FaceForensics) is a large-scale facial manipulation dataset generated using deepfake technologies. Each deepfake method is applied to 1000 high-quality videos uploaded to YouTube, manually selected to present subjects almost face-on without occlusion.

## Preprocessing
We apply the same data preprocessing strategy for the dataset:

- Extract the images from each video;
- Use reliable face tracking technology ([Blazeface](https://github.com/hollance/BlazeFace-PyTorch)) to detect faces in each frame and crop the image around the face.

When detecting videos created with face manipulation, it is possible to train the detectors on the entire frames of the videos, or simply crop the area around the face, and apply the detector exclusively to this cropped area. We applied the latter.

We used a series of different but simple data augmentation techniques. In particular, we randomly apply downscaling, horizontal flipping, random brightness contrast, hue saturation, noise, ... It should be noted here that data augmentation is only applied to training images.

## Model and train
In our experiments, we consider the EfficientNetB4 model as a baseline and we performed 4 configuration types :

- Configuration 1: Using the EfficicentNetB4 pre-trained model and freezing all convolution parameters.
- Configuration 2: Using the EfficicentNetB4 pre-trained model and re-training the last 8 MBConvBlocks with the last two layers (the convolution layer and the classifier).
- Configuration 3: Using the EfficicentNetB4 pre-trained model and retraining the last 16 MBConvBlocks with the last two layers (the convolution layer and the classifier).
- Configuration 4: Using the EfficicentNetB4 pre-trained model and integrating a Spatial Transformer Network after MBConvBloc 20.


