# import numpy as np
import pandas as pd
from glob import glob
from os.path import join
# from pathlib import Path
from PIL import Image
# import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
# import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.optim as optim

class AgeDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, annot_path, train=True):
        super(AgeDataset, self).__init__()

        self.annot_path = annot_path
        self.data_path = data_path
        self.train = train

        self.ann = pd.read_csv(annot_path)
        self.files = self.ann['file_id']
        if train:
            self.ages = self.ann['age']
        self.transform = self._transform(224)

    @staticmethod    
    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    def _transform(self, n_px):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return Compose([
            Resize(n_px),
            self._convert_image_to_rgb,
            ToTensor(),
            Normalize(mean, std),
        ])

    def read_img(self, file_name):
        im_path = join(self.data_path,file_name)   
        img = Image.open(im_path)
        img = self.transform(img)
        return img

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


train_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train'
train_ann = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train.csv'

# df = pd.read_csv(train_ann)
# strings_to_filter = ['image_10137.jpg', 'image_11592.jpg', 'image_12759.jpg', 'image_14036.jpg', 'image_15299.jpg', 'image_16477.jpg', 'image_18085.jpg', 'image_19381.jpg', 'image_20935.jpg', 'image_5595.jpg', 'image_7987.jpg', 'image_10224.jpg', 'image_11643.jpg', 'image_12822.jpg', 'image_14170.jpg', 'image_15306.jpg', 'image_16480.jpg', 'image_18087.jpg', 'image_19531.jpg', 'image_20967.jpg', 'image_5622.jpg', 'image_7994.jpg', 'image_10237.jpg', 'image_11668.jpg', 'image_12832.jpg', 'image_14226.jpg', 'image_15313.jpg', 'image_16552.jpg', 'image_18106.jpg', 'image_19611.jpg', 'image_20988.jpg', 'image_6074.jpg', 'image_8004.jpg', 'image_10441.jpg', 'image_11715.jpg', 'image_12869.jpg', 'image_14228.jpg', 'image_15317.jpg', 'image_16652.jpg', 'image_18177.jpg', 'image_20084.jpg', 'image_21056.jpg', 'image_6118.jpg', 'image_8104.jpg', 'image_10519.jpg', 'image_11797.jpg', 'image_12882.jpg', 'image_14229.jpg', 'image_15328.jpg', 'image_16848.jpg', 'image_18257.jpg', 'image_20118.jpg', 'image_21104.jpg', 'image_6438.jpg', 'image_8331.jpg', 'image_10536.jpg', 'image_11925.jpg', 'image_12897.jpg', 'image_14327.jpg', 'image_15427.jpg', 'image_16938.jpg', 'image_18295.jpg', 'image_20123.jpg', 'image_21184.jpg', 'image_6498.jpg', 'image_8348.jpg', 'image_10568.jpg', 'image_11963.jpg', 'image_12923.jpg', 'image_14410.jpg', 'image_1547.jpg', 'image_17043.jpg', 'image_18349.jpg', 'image_20129.jpg', 'image_21283.jpg', 'image_655.jpg', 'image_8397.jpg', 'image_10596.jpg', 'image_12057.jpg', 'image_13106.jpg', 'image_14513.jpg', 'image_1548.jpg', 'image_17117.jpg', 'image_18357.jpg', 'image_20157.jpg', 'image_21334.jpg', 'image_6675.jpg', 'image_8405.jpg', 'image_10773.jpg', 'image_12080.jpg', 'image_13123.jpg', 'image_14568.jpg', 'image_15522.jpg', 'image_17214.jpg', 'image_18394.jpg', 'image_20215.jpg', 'image_2164.jpg', 'image_6765.jpg', 'image_8493.jpg', 'image_10784.jpg', 'image_12082.jpg', 'image_13164.jpg', 'image_14736.jpg', 'image_15528.jpg', 'image_17240.jpg', 'image_1840.jpg', 'image_20321.jpg', 'image_2219.jpg', 'image_6807.jpg', 'image_8547.jpg', 'image_10792.jpg', 'image_12108.jpg', 'image_13257.jpg', 'image_14740.jpg', 'image_1561.jpg', 'image_17244.jpg', 'image_18446.jpg', 'image_20328.jpg', 'image_3202.jpg', 'image_6825.jpg', 'image_8735.jpg', 'image_10853.jpg', 'image_12114.jpg', 'image_13278.jpg', 'image_1475.jpg', 'image_15740.jpg', 'image_17293.jpg', 'image_18499.jpg', 'image_20400.jpg', 'image_3295.jpg', 'image_683.jpg', 'image_8766.jpg', 'image_10878.jpg', 'image_12168.jpg', 'image_13309.jpg', 'image_14778.jpg', 'image_15778.jpg', 'image_17371.jpg', 'image_18523.jpg', 'image_20406.jpg', 'image_3594.jpg', 'image_6854.jpg', 'image_8857.jpg', 'image_10917.jpg', 'image_12178.jpg', 'image_13314.jpg', 'image_1486.jpg', 'image_15848.jpg', 'image_1737.jpg', 'image_18557.jpg', 'image_20493.jpg', 'image_3609.jpg', 'image_6973.jpg', 'image_8986.jpg', 'image_10924.jpg', 'image_12183.jpg', 'image_13366.jpg', 'image_1487.jpg', 'image_15861.jpg', 'image_17455.jpg', 'image_18561.jpg', 'image_20506.jpg', 'image_3648.jpg', 'image_7047.jpg', 'image_9052.jpg', 'image_10935.jpg', 'image_12190.jpg', 'image_13429.jpg', 'image_14911.jpg', 'image_15954.jpg', 'image_1747.jpg', 'image_18567.jpg', 'image_20563.jpg', 'image_3739.jpg', 'image_7206.jpg', 'image_9065.jpg', 'image_10964.jpg', 'image_12302.jpg', 'image_13572.jpg', 'image_15031.jpg', 'image_16079.jpg', 'image_1754.jpg', 'image_18571.jpg', 'image_20579.jpg', 'image_3962.jpg', 'image_7234.jpg', 'image_910.jpg', 'image_10983.jpg', 'image_12332.jpg', 'image_1360.jpg', 'image_1503.jpg', 'image_16105.jpg', 'image_17587.jpg', 'image_18666.jpg', 'image_20594.jpg', 'image_4037.jpg', 'image_7257.jpg', 'image_9208.jpg', 'image_11113.jpg', 'image_12380.jpg', 'image_13618.jpg', 'image_15115.jpg', 'image_16126.jpg', 'image_17623.jpg', 'image_18779.jpg', 'image_20635.jpg', 'image_4210.jpg', 'image_7331.jpg', 'image_9502.jpg', 'image_11180.jpg', 'image_12446.jpg', 'image_13645.jpg', 'image_15141.jpg', 'image_16199.jpg', 'image_17664.jpg', 'image_18823.jpg', 'image_20652.jpg', 'image_4726.jpg', 'image_7345.jpg', 'image_9604.jpg', 'image_11246.jpg', 'image_12612.jpg', 'image_13648.jpg', 'image_15158.jpg', 'image_16202.jpg', 'image_17719.jpg', 'image_18873.jpg', 'image_20682.jpg', 'image_4930.jpg', 'image_7424.jpg', 'image_965.jpg', 'image_1133.jpg', 'image_12617.jpg', 'image_13751.jpg', 'image_15179.jpg', 'image_16302.jpg', 'image_17927.jpg', 'image_19004.jpg', 'image_20739.jpg', 'image_5076.jpg', 'image_7470.jpg', 'image_9712.jpg', 'image_11490.jpg', 'image_12638.jpg', 'image_13782.jpg', 'image_15228.jpg', 'image_16350.jpg', 'image_1794.jpg', 'image_19017.jpg', 'image_20784.jpg', 'image_528.jpg', 'image_7617.jpg', 'image_9814.jpg', 'image_11573.jpg', 'image_12723.jpg', 'image_13931.jpg', 'image_15234.jpg', 'image_16367.jpg', 'image_18018.jpg', 'image_19204.jpg', 'image_20786.jpg', 'image_5320.jpg', 'image_7727.jpg', 'image_9914.jpg', 'image_11577.jpg', 'image_12756.jpg', 'image_13984.jpg', 'image_15253.jpg', 'image_16376.jpg', 'image_18023.jpg', 'image_19378.jpg', 'image_20933.jpg', 'image_5384.jpg', 'image_7761.jpg']
# # len(strings_to_filter)
# # Filter the DataFrame
# df_filtered = df[~df['file_id'].isin(strings_to_filter)]
# # Save the filtered DataFrame back to a CSV
# df_filtered.to_csv('train1.csv', index=False)
# train_ann = 'train1.csv'

train_dataset = AgeDataset(train_path, train_ann, train=True)


test_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/test'
test_ann = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv'
test_dataset = AgeDataset(test_path, test_ann, train=False)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, criterion, optimizer, num_epochs, train_loss, modelName):
    model.train()  # Set model to training mode
    init = len(train_loss)
    for epoch in range(num_epochs):        
        running_loss = 0.0
        for images, ages in tqdm(train_loader):
            images, ages = images.to(device), ages.to(device).float()
            
            optimizer.zero_grad()
            
            outputs = model(images).squeeze()  # Squeeze to remove any extra dimensions
            loss = criterion(outputs, ages)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_loss.append(epoch_loss)
        print(f'Epoch {epoch + 1 + init}/{num_epochs + init}, Loss: {epoch_loss}')
    torch.save(model.state_dict(), f'AgePred{modelName}Mdl{len(train_loss)}Ep.pt')
    



###### SUBMISSION CSV FILE #####

@torch.no_grad
def predict(loader, model, init, modelName):
    model.eval()
    predictions = []

    for img in tqdm(loader):
        img = img.to(device)

        pred = model(img)
        predictions.extend(pred.flatten().detach().tolist())

#     return predictions

#     preds = predict(test_loader, model)

    submit = pd.read_csv('/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv')
#     submit['age'] = preds
    submit['age'] = predictions
    print()
    print(submit.head(20))
    print()
    submit.to_csv(f'baseline{modelName}Mdl{init}Ep.csv', index=False)

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
    model = torch.hub.load("pytorch/vision", f"{name}".lower(), pretrained=True) #weights = f"{name}_Weights.IMAGENET1K_V1", force_reload=True)
    
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
    
    # Load the pre-trained EfficientNet model
    # model = torch.hub.load('pytorch/vision', 'efficientnet_b0', weights='EfficientNet_B0_Weights.IMAGENET1K_V1', force_reload=True)
    # Replace the classifier layer
    # num_features = model.classifier[1].in_features  # Get the input features of the original classifier layer
    # model.classifier[1] = nn.Linear(num_features, 1)  # Replace it with a new linear layer

    # Load the EfficientNet_V2_M model
    # model = torch.hub.load('pytorch/vision', 'efficientnet_v2_s', weights='EfficientNet_V2_S_Weights.IMAGENET1K_V1')
    # Get the number of input features to the classifier
    # num_features = model.classifier[1].in_features
    # Replace the classifier with a new one with a single output feature (for binary classification)
    # model.classifier[1] = nn.Linear(num_features, 1)

    # Setup the loss function and optimizer
    model.to(device)
    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=7e-4, momentum=0.9, weight_decay=1e-3)


    
    # Train the model
    train_model(model, train_loader, criterion, optimizer, all_epochs[name], all_train_losses[name], name)
    predict(test_loader, model, len(all_train_losses[name]), name)




# Load the two CSV files into Pandas DataFrames
# df1 = pd.read_csv('rounded_ages_blah.csv')
# df2 = pd.read_csv('baseline_inception.csv')

# Merge the two DataFrames on the 'file_id' column
# merged_df = pd.merge(df1, df2, on='file_id', suffixes=('_1', '_2'))

# Calculate the average of the 'age' columns
# merged_df['age'] = ((merged_df['age_1'] + merged_df['age_2']) / 2 - 0.25)

# Select only the necessary columns for the new CSV
# result_df = merged_df[['file_id', 'age']]

# Save the result to a new CSV file
# result_df.to_csv('baseline.csv', index=False)


# List all CSV files in the current directory
file_list = glob('baseline*.csv')

# Read each CSV file and store in a list
dataframes = []
for file in file_list:
    df = pd.read_csv(file)
    dataframes.append(df)

# Concatenate all dataframes
combined_df = pd.concat(dataframes)

# Calculate average age for each file_id
average_age_df = combined_df.groupby('file_id')['age'].mean().reset_index()

# Save the result to a new CSV file
average_age_df.to_csv('average_ages.csv', index=False)

# Read the CSV file into a DataFrame
rounded_df = pd.read_csv('average_ages.csv')

# Round the 'age' column to integers
rounded_df['rounded_age'] = rounded_df['age'].round().astype(int)

# Create a new DataFrame with only 'file_id' and 'rounded_age' columns
new_df = rounded_df[['file_id', 'rounded_age']]

# Write the new DataFrame to a CSV file
new_df.to_csv('rounded_baseline.csv', index=False)
