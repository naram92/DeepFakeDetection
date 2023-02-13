# DeepFakeDetection
## Dataset
[FaceForensics++](https://github.com/ondyari/FaceForensics) is a large-scale facial manipulation dataset generated using deepfake technologies. Each deepfake method is applied to 1000 high-quality videos uploaded to YouTube, manually selected to present subjects almost face-on without occlusion.

## Preprocessing
We followed the same data preprocessing strategy for the dataset:

- Extract the images from each video;
- Use reliable face tracking technology ([Blazeface](https://github.com/hollance/BlazeFace-PyTorch)) to detect faces in each frame and crop the image around the face.

When detecting videos with manipulated faces, it is possible to train the detectors on the entire frames of the videos or to crop the area around the face and apply the detector exclusively to this cropped region. We chose the latter approach.

We applied a series of simple but effective data augmentation techniques, including downscaling, horizontal flipping, random brightness and contrast adjustments, hue and saturation changes, noise injection, ... It is worth noting that data augmentation was only applied to training images.

## Model and train
In our experiments, we used the [EfficientNetB4](https://github.com/lukemelas/EfficientNet-PyTorch) model as a baseline due to its favorable balance of number of parameters, runtime, and classification performance. We evaluated four configurations:

- Configuration 1: Using the EfficicentNetB4 pre-trained model and freezing all convolutional parameters.
- Configuration 2: Using the EfficicentNetB4 pre-trained model and retraining the last 8 MBConvBlocks with the last two layers (the convolutional layer and the classifier).
- Configuration 3: Using the EfficicentNetB4 pre-trained model and retraining the last 16 MBConvBlocks with the last two layers (the convolutional layer and the classifier).
- Configuration 4: Using the EfficicentNetB4 pre-trained model and integrating a Spatial Transformer Network after MBConvBloc 20.