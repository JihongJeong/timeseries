import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from MyModel import LSTMModel
import os
import pandas as pd
from data_preprocessing import load_data, plot_data, scale, inverse_scale, train_test_split, create_sequences, add_result

df, featueres = load_data()
plot_data(df)

df_scaled, scaler = scale(df)

X_train, y_train, X_test, y_test = train_test_split(df_scaled)

X_seq_train, y_seq_train = create_sequences(X_train, y_train, seq_length=30)
X_seq_test, y_seq_test = create_sequences(X_test, y_test, seq_length=30)

X_tensor = torch.tensor(X_seq_train, dtype=torch.float32)
y_tensor = torch.tensor(y_seq_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_seq_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_seq_test, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

testset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(testset, batch_size=32, shuffle=False)

lstm = LSTMModel(input_dim=X_tensor.shape[2], hidden_dim=10, output_dim=1,seq_len=X_tensor.shape[1])
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
train = True
epochs = 100

checkpoint = f"./checkpoints/train_with_{epochs}Epochs.pt"
if os.path.isfile(checkpoint):
    train = False

if train:
    for epoch in range(epochs):
        epoch_loss = 0
        
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = batch
            pred = lstm(x).squeeze()
            loss = loss_function(pred, y)
            epoch_loss += loss
            
            loss.backward()
            optimizer.step()
            if idx%10 == 0:
                print(f'Epoch(iteration) {epoch+1}/{epochs}({idx}/{len(train_loader)}), Loss: {loss}')
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader)}')

    torch.save(lstm.state_dict(), f"./checkpoints/train_with_{epochs}Epochs.pt")
else:
    lstm.load_state_dict(torch.load(checkpoint))
    
lstm.eval()
test_loss = 0
result = []
for idx, batch in enumerate(test_loader):
    x, y = batch
    pred = lstm(x).squeeze()
    result += pred.detach().tolist()
    test_loss += loss_function(pred, y)

print(f"Test Loss : {test_loss/len(test_loader)}")

result = np.array(result)
result = result.reshape(-1, 1)
result = inverse_scale(result.tolist(), scaler)

pred = add_result(df, result.squeeze().tolist())
plot_data(pred, with_prediction=True)