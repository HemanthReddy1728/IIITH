# Age Prediction Model Documentation

This document provides a detailed explanation of the Python code used for predicting ages from images using deep learning models. The code utilizes the PyTorch library along with several other Python libraries to preprocess data, train models, and predict ages.

## Prerequisites

Before running this script, ensure the following libraries are installed:

-   `numpy`
-   `pandas`
-   `PIL` (Pillow)
-   `torch`
-   `torchvision`
-   `tqdm`
-   `glob`

## Code Explanation

### Importing Libraries

The script begins by importing necessary Python libraries used throughout the code:

```python
import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.optim as optim
```

### Dataset Class

The `AgeDataset` class is a subclass of `torch.utils.data.Dataset` and is used to manage the dataset for training and testing purposes.

```python
class AgeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, annot_path, train=True):
        self.annot_path = annot_path
        self.data_path = data_path
        self.train = train
        self.ann = pd.read_csv(annot_path)
        self.files = self.ann['file_id']
        if train:
            self.ages = self.ann['age']
        self.transform = self._transform(224)

    def _transform(self, n_px):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return Compose([
            Resize(n_px),
            self._convert_image_to_rgb,
            ToTensor(),
            Normalize(mean, std),
        ])

    def __getitem__(self, index):
        file_name = self.files[index]
        img = self.read_img(file_name)
        if self.train:
            age = self.ages[index]
            return img, age
        else:
            return img

    def __len__(self):
        return len(self.files)
```

### Model Training and Prediction

The functions `train_model` and `predict` are defined to handle the training of the neural network and making predictions on new data, respectively.

```python
def train_model(model, train_loader, criterion, optimizer, num_epochs, train_loss, modelName):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, ages in tqdm(train_loader):
            images, ages = images.to(device), ages.to(device).float()
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        train_loss.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}')
    torch.save(model.state_dict(), f'AgePred{modelName}Mdl{len(train_loss)}Ep.pt')

@torch.no_grad()
def predict(loader, model, init, modelName):
    model.eval()
    predictions = []
    for img in tqdm(loader):
        img = img.to(device)
        pred = model(img)
        predictions.extend(pred.flatten().detach().tolist())
    submit = pd.read_csv('/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv')
    submit['age'] = predictions
    submit.to_csv(f'baseline{modelName}Mdl{init}Ep.csv', index=False)
```

### Model Initialization and Setup

```python
modelNames = [
    'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'ConvNeXt_Base'
    'EfficientNet_B0', 'EfficientNet_B1', 'EfficientNet_B2', 'EfficientNet_B3', 'EfficientNet_B4', 'EfficientNet_B5', 'EfficientNet_B6', 'EfficientNet_B7',
    'EfficientNet_V2_S', 'EfficientNet_V2_M',
    'MobileNet_V3_Large', 'Inception_V3', 'MaxVit_T',
    'Swin_S', 'Swin_T', 'Swin_V2_S', 'Swin_V2_T',
    'ViT_B_16', 'ViT_B_32'
    ]
all_train_losses = {name : [] for name in modelNames}
all_epochs = {name : 30 for name in modelNames}
for name in modelNames:
    model = torch.hub.load("pytorch/vision", f"{name}".lower(), pretrained=True)

    nameIndex = modelNames.index(name)
    if nameIndex in range(6):
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)
    elif nameIndex in range(6, 16):
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 1)
    elif nameIndex == 16:
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_features, 1)
    elif nameIndex == 17:
        model.aux_logits = True
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)
        num_aux_features = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_aux_features, 1)
    elif nameIndex in range(18, 25):
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, 1)

    # Setup the loss function and optimizer
    model.to(device)
    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=7e-4, momentum=0.9, weight_decay=1e-3)
    train_model(model, train_loader, criterion, optimizer, all_epochs[name], all_train_losses[name], name)
    predict(test_loader, model, len(all_train_losses[name]), name)
```

### File Operations and Data Aggregation via Ensemble Analysis by Averaging and Rounding

After models have made predictions, the results are processed to combine and calculate the average age predictions:

```python
file_list = glob('baseline*.csv')
dataframes = []
for file in file_list:
    df = pd.read_csv(file)
    dataframes.append(df)

combined_df = pd.concat(dataframes)
average_age_df = combined_df.groupby('file_id')['age'].mean().reset_index()
average_age_df.to_csv('average_ages.csv', index=False)

rounded_df = pd.read_csv('average_ages.csv')
rounded_df['rounded_age'] = rounded_df['age'].round().astype(int)
new_df = rounded_df[['file_id', 'rounded_age']]
new_df.to_csv('rounded_baseline.csv', index=False)
```

## Detailed Overview of Deep Learning Models

The following descrption is a comprehensive overview of various deep learning models used in the context of age prediction from images. Each model is briefly introduced with its architectural details and key characteristics. This overview can help in understanding the strengths and use cases of each model for specific applications.

### ResNet (Residual Networks)

Residual Networks, or ResNet, introduced by Microsoft Research, are a family of models designed to address the degradation problem in very deep networks. By using skip connections or shortcuts to jump over some layers, ResNets allow for training deeper networks by enabling feature reusability and preventing the vanishing gradient problem.

- **ResNet18 and ResNet34**: Smaller versions with 18 and 34 layers, respectively. They are ideal for datasets with lower complexity and environments with stringent computational or memory constraints.
  
- **ResNet50, ResNet101, and ResNet152**: These versions have 50, 101, and 152 layers, respectively. They are more suitable for complex datasets due to their deeper architectures, which can learn more detailed features.

### ConvNeXt

ConvNeXt models are an evolution of the traditional convolutional networks, integrating some of the design principles from Transformers. They provide a more uniform scaling strategy and have improved performance over standard convolutional bases.

- **ConvNeXt_Base**: A baseline model of the ConvNeXt family, which typically provides a good balance between performance and computational efficiency, making it suitable for a variety of image recognition tasks.

### EfficientNet

EfficientNet models are a family of convolutional neural networks designed to achieve state-of-the-art accuracy with significantly fewer parameters and lower computational cost. The architecture uses a compound scaling method to uniformly scale the depth, width, and resolution of the network.

- **EfficientNet_B0 to EfficientNet_B7**: This series from B0 to B7 represents progressively larger networks, where each subsequent model is roughly 15% more computationally expensive than the previous one. These models provide a spectrum of capabilities, allowing for tailored solutions based on the computational resources and the complexity of the task.

### EfficientNet_V2

The EfficientNet_V2 models are newer versions that focus on improving training speed and parameter efficiency. These models use a combination of techniques from the original EfficientNet design and newer advancements such as progressive learning.

- **EfficientNet_V2_S and EfficientNet_V2_M**: These are small and medium variants of the EfficientNet_V2 series, designed to offer better performance and efficiency, especially in terms of faster training times.

### MobileNet_V3

MobileNet_V3 models are part of the MobileNet family, optimized for mobile and edge devices. They utilize lightweight depthwise separable convolutions and incorporate hardware-aware optimization to enhance performance.

- **MobileNet_V3_Large**: This variant offers a good trade-off between latency and accuracy, making it well-suited for real-time applications on mobile devices.

### Inception_V3

Inception_V3 is a convolutional neural network that is 48 layers deep. It uses multiple sized convolutional filters within the same layer to gather information from various scales concurrently. This model is known for its high accuracy in classifying images.

- **Inception_V3**: Utilizes factorized convolutions and aggressive regularization strategies to improve training speed and reduce overfitting.

### MaxVit

MaxVit models are a recent addition that combine the strengths of Vision Transformers (ViT) and MaxPooling, enabling robust spatial feature extraction and efficient model scaling.

- **MaxVit_T**: A smaller version that can provide Transformer-like capabilities but is more manageable in terms of computational demand, suitable for a range of vision tasks.

### Swin Transformer

Swin Transformers are a type of Vision Transformer that uses shifted windows to limit self-attention computation to non-overlapping local windows while also allowing for cross-window connection.

- **Swin_S and Swin_T**: These are small and tiny versions of the Swin Transformer, designed to be efficient while still capturing complex hierarchical features.
- **Swin_V2_S and Swin_V2_T**: Improved versions of the original Swin models that enhance efficiency and accuracy, adapting dynamically to different computational and memory requirements.

### Vision Transformer (ViT)

ViT applies the Transformer architecture directly to sequences of image patches, treating them as tokens, akin to words in text processing.

- **ViT_B_16 and ViT_B_32**: These models differ in the size of the image patches they process, with "16" and "32" referring to the dimensions of each patch. ViT_B_16 processes smaller patches and therefore, captures more detailed information, making it more suitable for higher-resolution images.

## Conclusion

The selection of a model often depends on the specific requirements and constraints of the application, such as computational resources, latency requirements, and the complexity of the task. In the age prediction project, these models are explored to determine which provides the best balance of accuracy and efficiency. This documentation also provides a comprehensive overview of the code used for age prediction. The detailed descriptions of each part of the code help in understanding how the model is set up, trained, and used for making predictions on new data.
The models described provide a broad range of tools for tackling different image recognition tasks, from mobile applications requiring real-time processing to complex problems demanding high-accuracy solutions. By understanding the characteristics of each model, developers can better choose the right model based on the specific needs and constraints of their application.
