import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from MyModel import LSTMModel
import os
import pandas as pd
from data_preprocessing import load_data, plot_data, scale, inverse_scale, train_test_split, create_sequences, add_result

parser = argparse.ArgumentParser()
parser.add_argument('-m', dest = 'mode', action = 'store', default = 'train/test', help = 'available modes : train/test, inference')
parser.add_argument('-lp', dest = 'latest_price', type = float, action = 'store', help = 'latest price to decide action')
info = parser.parse_args()

if info.mode == 'train/test':
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

elif info.mode == 'inference':
    df, featueres = load_data(all = False)
   
    df_scaled, scaler = scale(df)
    X = torch.tensor(df_scaled[-30:].reshape(1, 30, -1), dtype = torch.float32)
    
    lstm = LSTMModel(input_dim=X.shape[2], hidden_dim=10, output_dim=1,seq_len=X.shape[1])
    checkpoint = f"./checkpoints/train_with_100Epochs.pt"
    lstm.load_state_dict(torch.load(checkpoint))
    pred = lstm(X)
    
    result = pred.reshape(-1, 1).detach()
    result = inverse_scale(result.tolist(), scaler)
    
    prev = info.latest_price if info.latest_price else df['Close'].values[-1][0]
    pred = result[0][0]
    
    change = (pred - prev)/prev * 100
    if change > 2:
        action = 'Buy'
    elif change < -2:
        action = 'Sell'
    else:
        action = 'Hold'
    
    print(f"{action}, predicted increase/decrease rate is {change:.2f}%")
    
else:
    print("Wrong parser!")
