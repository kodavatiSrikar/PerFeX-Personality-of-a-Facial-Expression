import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from torchsummary import summary
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
        x = max(self.au_data['ID'])
        y = min(self.au_data['ID'])
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
        return torch.tensor(au).float(), torch.tensor(length).int(), torch.tensor(personality).float(), self.name_list[idx]

# Define the function to create the source mask
def create_src_mask(lengths, max_length):
    mask = torch.arange(max_length, device=lengths.device)[None, :] >= lengths[:, None]
    return mask

# Load the model checkpoint
model = PersonalityPredictionModel(au_dim=17, lstm_hidden_size=128, max_seq_length=700).to(device)

model.load_state_dict(torch.load('cnn_180.pt'))  # replace with your checkpoint file
model.eval()

# Create dataset and data loader
test_dataset = FaceDataset('final_created.csv')  # replace with your test dataset file
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Perform prediction
with torch.no_grad():
    pred_dict={}
    for action_units, lengths, personalities,ID in test_loader:
        action_units, lengths, personalities = action_units.to(device), lengths.to(device), personalities.to(device)
        timesteps = None
        output = model(action_units, lengths, timesteps)
        y_pred1 = torch.softmax(output[:, 0:5], dim=1).argmax(dim=1)
        y_pred2 = torch.softmax(output[:, 5:10], dim=1).argmax(dim=1)
        y_pred3 = torch.softmax(output[:, 10:15], dim=1).argmax(dim=1)
        y_pred4 = torch.softmax(output[:, 15:20], dim=1).argmax(dim=1)
        y_pred5 = torch.softmax(output[:, 20:25], dim=1).argmax(dim=1)
        pred_dict[int(ID[0])] = {'extroversion': y_pred1/4,
                            'neuroticism': y_pred2/4,
                            'agreeableness': y_pred3/4,
                            'conscientiousness': y_pred4/4,
                            'openness': y_pred5/4,
                                  }
    au_data = pd.read_csv('final_created.csv')
    for i in pred_dict.keys():
        print(i)
        label_out=pred_dict[i]
        au_data.loc[au_data['ID'] == i, 'extroversion'] = float(label_out['extroversion'][0])
        au_data.loc[au_data['ID'] == i, 'neuroticism'] = float(label_out['neuroticism'][0])
        au_data.loc[au_data['ID'] == i, 'agreeableness'] = float(label_out['agreeableness'][0])
        au_data.loc[au_data['ID'] == i, 'conscientiousness'] = float(label_out['conscientiousness'][0])
        au_data.loc[au_data['ID'] == i, 'openness'] = float(label_out['openness'][0])

    print(au_data.head(15))
    au_data=au_data.dropna()
    au_data.to_csv('final_pers.csv',index=False)

        
        