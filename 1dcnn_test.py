import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the model
class PersonalityPredictionModel(nn.Module):
    def __init__(self, au_dim, lstm_hidden_size, max_seq_length=700, latent_dim=128, max_nb_variables=17, max_timesteps=700, nb_classes=25):
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

# Define the FaceDataset class
class FaceDataset(Dataset):
    def __init__(self, au_file, max_length=700):
        self.au_data = pd.read_csv(au_file)
        self.max_length = max_length
        dataset_complete = self.au_data.drop(['Word','vi_1','vi_2','vi_3','vi_4','vi_5'], axis=1)
        x = max(dataset_complete['ID'])
        y = min(dataset_complete['ID'])
        print(y,x)
        data_dict = {}
        new_name_list = []
        for i in range(y, x+1):
            dataset = dataset_complete[dataset_complete['ID'] == i]
            if (len(dataset) < self.max_length) and (len(dataset) > 40):
                key = dataset['File'].iloc[0]
                key = key.replace('.mp4', '')
                dataset_temp = dataset[dataset.columns.drop(list(dataset.filter(regex='_')))]
                dataset1 = dataset.drop(['ID', 'File'], axis=1)
                dataset_out = dataset_temp.drop(['ID', 'File','Happy','Angry','Surprise','Sad','Fear'], axis=1)
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
        return torch.tensor(au).float(), torch.tensor(length).int(), torch.tensor(personality).float()

# Define the function to create the source mask
def create_src_mask(lengths, max_length):
    mask = torch.arange(max_length, device=lengths.device)[None, :] >= lengths[:, None]
    return mask

# Load the model checkpoint
model = PersonalityPredictionModel(au_dim=17, lstm_hidden_size=128, max_seq_length=700).to(device)
model.load_state_dict(torch.load('cnn_180.pt'))  # replace with your checkpoint file
model.eval()

# Create dataset and data loader
test_dataset = FaceDataset('data_range.csv')  # replace with your test dataset file
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Perform prediction
with torch.no_grad():
    y_true = [[] for _ in range(5)]
    y_pred = [[] for _ in range(5)]
    y_probs = [[] for _ in range(5)]
    for action_units, lengths, personalities in test_loader:
        action_units, lengths, personalities = action_units.to(device), lengths.to(device), personalities.to(device)
        timesteps = None
        output = model(action_units, lengths, timesteps)
        y_prob1=torch.softmax(output[:, 0:5], dim=1)
        y_pred1 = y_prob1.argmax(dim=1)
        y_probs[0].append(y_prob1.cpu().numpy())
        y_pred[0].append(y_pred1.cpu().numpy())
        y_true[0].append((personalities[0][0] * 4).long().cpu().numpy())
        y_prob2= torch.softmax(output[:, 5:10], dim=1)
        y_pred2 = y_prob2.argmax(dim=1)
        y_probs[1].append(y_prob2.cpu().numpy())
        y_pred[1].append(y_pred2.cpu().numpy())
        y_true[1].append((personalities[0][1] * 4).long().cpu().numpy())
        y_prob3 = torch.softmax(output[:, 10:15], dim=1)
        y_pred3 = y_prob3.argmax(dim=1)
        y_probs[2].append(y_prob3.cpu().numpy())
        y_pred[2].append(y_pred3.cpu().numpy())
        y_true[2].append((personalities[0][2] * 4).long().cpu().numpy())
        y_prob4 = torch.softmax(output[:, 15:20], dim=1)
        y_pred4 = y_prob4.argmax(dim=1)
        y_probs[3].append(y_prob4.cpu().numpy())
        y_pred[3].append(y_pred4.cpu().numpy())
        y_true[3].append((personalities[0][3] * 4).long().cpu().numpy())
        y_prob5 = torch.softmax(output[:, 20:25], dim=1)
        y_pred5 = y_prob5.argmax(dim=1)
        y_probs[4].append(y_prob5.cpu().numpy())
        y_pred[4].append(y_pred5.cpu().numpy())
        y_true[4].append((personalities[0][4] * 4).long().cpu().numpy())

y_pred = [np.concatenate(trait) for trait in y_pred]
y_probs = [np.concatenate(trait, axis=0) for trait in y_probs]

for i in range(5):
    y_true_binarized = label_binarize(y_true[i], classes=[0, 1, 2, 3, 4])
    accuracy = accuracy_score(y_true[i], y_pred[i])
    precision = precision_score(y_true[i], y_pred[i], average='macro')
    recall = recall_score(y_true[i], y_pred[i], average='macro')
    roc_auc = roc_auc_score(y_true_binarized, y_probs[i], multi_class='ovr', average='macro')

    print(f'Trait {i+1}:')
    print(f'  Accuracy: {accuracy:.4f}')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall: {recall:.4f}')
    print(f'  ROC AUC: {roc_auc:.4f}')