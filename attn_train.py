import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

def create_timesteps(num_timesteps, scale=1.0):
    """
    Create timesteps for DDIM based on a specific range and scale.

    :param num_timesteps: The number of timesteps.
    :param scale: The scaling factor for the timesteps.
    :return: A 1-D tensor of timesteps.
    """
    # Generate timesteps in the range [0, num_timesteps) and scale them
    timesteps = torch.arange(0, num_timesteps, dtype=torch.float32)*scale
    return timesteps
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
def create_src_mask(lengths, max_length):
    mask = torch.arange(max_length, device=lengths.device)[None, :] >= lengths[:, None]
    return mask


class SelfAttention(nn.Module):
    def __init__(self, input_dim, alength):
        super(SelfAttention, self).__init__()
        self.alength = alength
        self.W1 = nn.Parameter(torch.randn(alength, input_dim))
        self.W2 = nn.Parameter(torch.randn(input_dim, alength))

    def forward(self, inputs):
        hidden_states_transposed = inputs.permute(0, 2, 1)
        attention_score = torch.matmul(self.W1, hidden_states_transposed)
        attention_score = torch.tanh(attention_score)
        attention_weights = torch.matmul(self.W2, attention_score)
        attention_weights = torch.softmax(attention_weights, dim=2)
        embedding_matrix = torch.matmul(attention_weights, inputs)
        return embedding_matrix

# Define the LSTM with Self-Attention model
class LSTMWithSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_length, alength):
        super(LSTMWithSelfAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.self_attn = SelfAttention(hidden_dim, alength)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.seq_length = seq_length

    def forward(self, x, src_mask=None):
      # Assuming x is a single sample, add batch dimension
        # Pass through the LSTM layer
        lstm_output, _ = self.lstm(x)

        # Assuming you want the final hidden state as output
        lstm_output = lstm_output[:, -1, :]

        attn_output = self.self_attn(lstm_output.unsqueeze(0))
        if src_mask is not None:
            attn_output = attn_output.masked_fill(src_mask.unsqueeze(-1), -float('inf'))
        x = self.output_layer(attn_output)
        return x


# Define the Squeeze-and-Excite block
class SqueezeExciteBlock(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(SqueezeExciteBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // reduction_ratio)
        self.fc2 = nn.Linear(input_channels // reduction_ratio, input_channels)
        
    def forward(self, x):
        batch_size, channels, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(batch_size, channels, 1)
        return x * y.expand_as(x)





# Define the model
class PersonalityPredictionModel(nn.Module):
    def __init__(self, au_dim, lstm_hidden_size, max_seq_length=700, latent_dim=512, max_nb_variables=17, max_timesteps=700, nb_classes=25):
        super(PersonalityPredictionModel, self).__init__()
        self.max_seq_length = max_seq_length
        self.latent_dim = latent_dim
        self.time_embed_dim = latent_dim
        self.au_dim = au_dim
        self.au_emb = nn.Linear(self.au_dim, self.latent_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.lstm = nn.LSTM(input_size=self.latent_dim, hidden_size=lstm_hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=lstm_hidden_size, num_heads=1)
        stride = 3
        self.attention_lstm = LSTMWithSelfAttention(input_dim=self.latent_dim, hidden_dim=256, output_dim=1, seq_length=max_timesteps, alength=max_timesteps)
        self.dropout = nn.Dropout(0.5)

        self.conv1d_y1 = nn.Conv1d(max_nb_variables, 128, 8, padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv1d_y2 = nn.Conv1d(128, 256, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv1d_y3 = nn.Conv1d(256, 256, 3, padding=1)
        # self.bn3 = nn.BatchNorm1d(128)

        self.conv1d_y4 = nn.Conv1d(256, 256, 3, padding=1)
        # self.bn4 = nn.BatchNorm1d(256)

        self.squeeze_excite1 = SqueezeExciteBlock(128)
        self.squeeze_excite2 = SqueezeExciteBlock(256)
        self.squeeze_excite3 = SqueezeExciteBlock(256)
        self.squeeze_excite4 = SqueezeExciteBlock(256)

        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            
            nn.Linear(in_features=256, out_features=128),
            nn.Linear(in_features=128, out_features=nb_classes), # how many classes are there?
        )

    def forward(self, action_units, lengths, timesteps):
        # Create source mask
        src_mask = create_src_mask(lengths, self.max_seq_length).unsqueeze(-1)

        # Expand src_mask to match the shape of action_units
        src_mask = src_mask.expand(-1, -1, action_units.size(-1))

        # Apply source mask
        action_units_masked = action_units.masked_fill(src_mask, 0)

        # Embed the masked action units
        packed_emb = self.au_emb(action_units_masked)

        # Add time embeddings
        time_enc = timestep_embedding(timesteps, self.latent_dim)
        time_emb = self.time_embed(time_enc)
        packed_emb = packed_emb + time_emb

        # Pass through the attention LSTM
        x = self.attention_lstm(packed_emb)
        x = self.dropout(x)

        y = action_units.permute(0, 2, 1)
        sr_mask=src_mask.permute(0,2,1)
        y = y.masked_fill(sr_mask, 0)
        y = F.relu(self.conv1d_y1(y))
        y = self.squeeze_excite1(y)
        y = self.dropout(y)
        y = F.relu(self.conv1d_y2(y))
        y = self.squeeze_excite2(y)
        y = F.relu(self.conv1d_y3(y))
        y = self.squeeze_excite3(y)
        y = self.dropout(y)
        y = F.relu(self.conv1d_y4(y))
        y = self.squeeze_excite4(y)

        # Apply global average pooling to reduce the dimensions of y
        y = self.global_avg_pooling(y)

        # Reshape y to match the number of features in x before concatenation
        
        x = torch.cat((x.squeeze(-1), y.squeeze(-1)), dim=1)
        x = self.linear_layer_stack(x.squeeze(-1))
        return x




class FaceDataset(Dataset):
    def __init__(self, au_file, max_length=700):
        self.au_data = pd.read_csv(au_file)
        self.max_length = max_length
        dataset_complete = self.au_data.drop(['Word','vi_1','vi_2','vi_3','vi_4','vi_5'], axis=1)
        x = max(dataset_complete['ID'])
        y = min(dataset_complete['ID'])
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


def build_dataloader(dataset, batch_size, train_ratio=0.8, shuffle=True):
    # Calculate the number of training and test samples
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset into training and test sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader for training and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataset and data loader
train_dataset = FaceDataset('data_augumented.csv')
train_loader, test_loader = build_dataloader(train_dataset, batch_size=1, shuffle=True)

# Initialize the model, optimizer, and criterion
model = PersonalityPredictionModel(au_dim=17, lstm_hidden_size=128, max_seq_length=700).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()




# Training loop for each video
# Training loop
num_epochs = 200
print_interval = 1000
save_interval = 5

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (action_units, lengths, personalities) in enumerate(train_loader):
        action_units, lengths, personalities = action_units.to(device), lengths.to(device), personalities.to(device)
        num_timesteps = 700
        scale = 100/30
        timesteps = create_timesteps(num_timesteps, scale).to(device)
        optimizer.zero_grad()
        output = model(action_units, lengths, timesteps)
        
        
        loss1 = criterion(output[:, 0:5], torch.tensor([int(personalities[0][0]*4)]).to(device))
        loss2 = criterion(output[:, 5:10], torch.tensor([int(personalities[0][1]*4)]).to(device))
        loss3 = criterion(output[:, 10:15], torch.tensor([int(personalities[0][2]*4)]).to(device))
        loss4 = criterion(output[:, 15:20], torch.tensor([int(personalities[0][3]*4)]).to(device))
        loss5 = criterion(output[:, 20:25], torch.tensor([int(personalities[0][4]*4)]).to(device))

        # Sum up the losses
        
        # Perform backward pass for each loss separately
        loss5.backward(retain_graph=True)
        loss4.backward(retain_graph=True)
        loss3.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        loss1.backward()

        optimizer.step()

        # Sum up the losses for logging purposes
        total_loss = loss1 + loss2 + loss3 + loss4 + loss5

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

    

                
                correct1 += (y_pred1/4 == personalities[0][0].item())
                correct2 += (y_pred2/4 == personalities[0][1].item())
                correct3 += (y_pred3/4 == personalities[0][2].item())
                correct4 += (y_pred4/4 == personalities[0][3].item())
                correct5 += (y_pred5/4 == personalities[0][4].item())
                total += personalities.size(0)
            accuracy = correct1 / total
            print(accuracy,correct2 / total,correct3 / total,correct4 / total,correct5 / total, '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            

            correct1 = 0
            correct2 = 0
            correct3 = 0
            correct4 = 0
            correct5 = 0
            total = 0
            for action_units, lengths, personalities in test_loader:
                action_units, lengths, personalities = action_units.to(device), lengths.to(device), personalities.to(device)
                output = model(action_units, lengths, timesteps)
                y_pred1 = torch.softmax(output[:, 0:5], dim=1).argmax(dim=1)
                y_pred2 = torch.softmax(output[:, 5:10], dim=1).argmax(dim=1)
                y_pred3 = torch.softmax(output[:, 10:15], dim=1).argmax(dim=1)
                y_pred4 = torch.softmax(output[:, 15:20], dim=1).argmax(dim=1)
                y_pred5 = torch.softmax(output[:, 20:25], dim=1).argmax(dim=1)

    

                
                correct1 += (y_pred1/4 == personalities[0][0].item())
                correct2 += (y_pred2/4 == personalities[0][1].item())
                correct3 += (y_pred3/4 == personalities[0][2].item())
                correct4 += (y_pred4/4 == personalities[0][3].item())
                correct5 += (y_pred5/4 == personalities[0][4].item())
                total += personalities.size(0)
            accuracy = correct1 / total
            print(accuracy,correct2 / total,correct3 / total,correct4 / total,correct5 / total, '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # Save model
        torch.save(model.state_dict(), f"attn_{epoch}.pt")

