import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torch.nn.functional as F
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torch.utils.data import DataLoader, random_split
def create_src_mask(lengths, max_length):
    mask = torch.arange(max_length, device=lengths.device)[None, :] >= lengths[:, None]
    return mask


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


# Define the model
class PersonalityPredictionModel(nn.Module):
    def __init__(self, au_dim, max_seq_length=700, latent_dim=128, max_nb_variables=17, max_timesteps=700, nb_classes=25):
        super(PersonalityPredictionModel, self).__init__()
        self.max_seq_length = max_seq_length
        self.latent_dim = latent_dim
        self.au_dim = au_dim

        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=128*4, out_features=128),
            nn.Linear(in_features=128, out_features=nb_classes)
        )
        
        self.conv1_1 = nn.Conv1d(max_nb_variables, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv1d(max_nb_variables, 64, kernel_size=5, padding=2)
        self.conv2_2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3_1 = nn.Conv1d(max_nb_variables, 64, kernel_size=7, padding=3)
        self.conv3_2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.conv4_1 = nn.Conv1d(max_nb_variables, 64, kernel_size=9, padding=4)
        self.conv4_2 = nn.Conv1d(64, 128, kernel_size=9, padding=4)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv1_drop = nn.Dropout(0.5)
        self.conv2_drop = nn.Dropout(0.5)
        self.conv3_drop = nn.Dropout(0.5)
        self.conv4_drop = nn.Dropout(0.5)

    def forward(self, action_units, lengths, timesteps):
        src_mask = create_src_mask(lengths, self.max_seq_length).unsqueeze(-1)
        src_mask = src_mask.expand(-1, -1, action_units.size(-1))
        y = action_units.permute(0, 2, 1)
        sr_mask = src_mask.permute(0, 2, 1)
        y = y.masked_fill(sr_mask, 0)

        x1 = self.relu(self.conv1_1(y))
        x1 = self.relu(self.conv1_drop(self.conv1_2(x1)))
        x2 = self.relu(self.conv2_1(y))
        x2 = self.relu(self.conv2_drop(self.conv2_2(x2)))
        x3 = self.relu(self.conv3_1(y))
        x3 = self.relu(self.conv3_drop(self.conv3_2(x3)))
        x4 = self.relu(self.conv4_1(y))
        x4 = self.relu(self.conv4_drop(self.conv4_2(x4)))
        y = torch.cat((x1, x2, x3, x4), dim=1)
        y = self.global_avg_pooling(y)
        x = self.linear_layer_stack(y.squeeze(-1))
        return x



class FaceDataset(Dataset):
    def __init__(self, au_file, max_length=700):
        self.au_data = pd.read_csv(au_file)
        self.max_length = max_length
        # dataset_complete = self.au_data.drop(['Word','vi_1','vi_2','vi_3','vi_4','vi_5'], axis=1)
        x = max(self.au_data['ID'])
        y = min(self.au_data['ID'])
        print(x)
        data_dict = {}
        new_name_list = []
        for i in range(y, x+1):
            dataset = self.au_data[self.au_data['ID'] == i]
            if (len(dataset) < self.max_length) and (len(dataset) > 40):
                key = dataset['File'].iloc[0]
                key = key.replace('.mp4', '')
                dataset_temp = dataset[dataset.columns.drop(list(dataset.filter(regex='_')))]
                dataset1 = dataset.drop(['ID', 'File'], axis=1)
                dataset_out = dataset_temp.drop(['ID', 'File'], axis=1)
                dataset_au = dataset1.filter(regex='_r', axis=1)
                values_out = dataset_out.iloc[0].values
                values_au = dataset_au.values
                data_dict[i] = {'AU': values_au,
                                  'length': len(values_au),
                                  'Personality': values_out}
                new_name_list.append(i)

        self.data_dict = data_dict
        print(len(self.data_dict))
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        label = self.data_dict[self.name_list[idx]]
        au = label['AU']
        length = label['length']
        personality = label['Personality']
        if length < self.max_length:
            padding_len = self.max_length - length
            D = au.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            au = np.concatenate((au, padding_zeros), axis=0)
            # print(au)
            # print(personality)
        return torch.tensor(au).float(), torch.tensor(length).int(), torch.tensor(personality, dtype=torch.float32)

# def build_dataloader(dataset, batch_size, shuffle=True):
#     data_loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#     )
#     return data_loader

def build_dataloader(dataset, batch_size, train_ratio=1, shuffle=False):
    # Calculate the number of training and test samples
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    print(train_size,test_size)
    # Split the dataset into training and test sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(len(train_dataset))
    # Create DataLoader for training and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# Create dataset and data loader
train_dataset = FaceDataset('retrain_final.csv')
train_loader, test_loader = build_dataloader(train_dataset, batch_size=1, shuffle=True)

# Initialize the model, optimizer, and criterion
model = PersonalityPredictionModel(au_dim=17, max_seq_length=700).to(device)
model.load_state_dict(torch.load("cnn_145.pt"))
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()




# Training loop for each video
# Training loop
num_epochs = 100
print_interval = 1000
save_interval = 5

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (action_units, lengths, personalities) in enumerate(train_loader):
        action_units, lengths, personalities = action_units.to(device), lengths.to(device), personalities.to(device)
        num_timesteps = 700
        scale = 100/30
        timesteps = None
        optimizer.zero_grad()
        torch.cuda.synchronize()
        output = model(action_units, lengths, timesteps)
        torch.cuda.synchronize()
        # print(output1,personalities)
        # print(torch.tensor([int(personalities[0][0]*4)]))
        
        loss1 = criterion(output[:, 0:5], torch.tensor([int(personalities[0][0]*4)]).to(device))
        loss2 = criterion(output[:, 5:10], torch.tensor([int(personalities[0][1]*4)]).to(device))
        loss3 = criterion(output[:, 10:15], torch.tensor([int(personalities[0][2]*4)]).to(device))
        loss4 = criterion(output[:, 15:20], torch.tensor([int(personalities[0][3]*4)]).to(device))
        loss5 = criterion(output[:, 20:25], torch.tensor([int(personalities[0][4]*4)]).to(device))

        # Sum up the losses
        total_loss = loss1+loss2+loss3+loss4+loss5
        total_loss.backward()
        optimizer.step()
        if batch_idx % print_interval == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item()}")
    
    # Calculate MAPE every 5 epochs
    if epoch % save_interval == 0:
        with torch.no_grad():
            model.eval()
            correct1 = 0
            correct2 = 0
            correct3 = 0
            correct4 = 0
            correct5 = 0
            total = 0
            for action_units, lengths, personalities in train_loader:
                action_units, lengths, personalities = action_units.to(device), lengths.to(device), personalities.to(device)
                output = model(action_units, lengths, timesteps)
                y_pred1 = torch.softmax(output[:, 0:5], dim=1).argmax(dim=1)
                y_pred2 = torch.softmax(output[:, 5:10], dim=1).argmax(dim=1)
                y_pred3 = torch.softmax(output[:, 10:15], dim=1).argmax(dim=1)
                y_pred4 = torch.softmax(output[:, 15:20], dim=1).argmax(dim=1)
                y_pred5 = torch.softmax(output[:, 20:25], dim=1).argmax(dim=1)

    

                # predictions = torch.tensor([[output1.argmax(dim=1)/4,output2.argmax(dim=1)/4,output3.argmax(dim=1)/4,output4.argmax(dim=1)/4,output5.argmax(dim=1)/4]]).to(device)
                # print(y_pred1/4,personalities[0][0])
                correct1 += (y_pred1/4 == personalities[0][0].item())
                correct2 += (y_pred2/4 == personalities[0][1].item())
                correct3 += (y_pred3/4 == personalities[0][2].item())
                correct4 += (y_pred4/4 == personalities[0][3].item())
                correct5 += (y_pred5/4 == personalities[0][4].item())
                total += personalities.size(0)
            accuracy = correct1 / total
            print(accuracy,correct2 / total,correct3 / total,correct4 / total,correct5 / total, '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            

                  # Save model
        print(epoch+150)
        torch.save(model.state_dict(), f"cnn_{epoch+150}.pt")

