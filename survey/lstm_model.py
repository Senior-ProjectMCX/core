import os
import argparse
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
torch.manual_seed(42)

class AutocorrelationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DCSLSTMModel(nn.Module):
    def __init__(self, input_size=167, hidden_size=512, num_layers=2, output_size=2, dropout=0.1):
        super(DCSLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def preprocess_and_split(mat_path, sds_channel='all'):
    mat = scipy.io.loadmat(mat_path)
    gtau = mat['ave_gtau_intra40']
    preset_s = gtau[3, 0, :]
    preset_d = gtau[4, 0, :]
    measured_s = gtau[5, 0, :]
    measured_d = gtau[6, 0, :]
    combos = np.vstack((preset_s, preset_d)).T
    unique_combos, inverse_indices = np.unique(combos, axis=0, return_inverse=True)
    train_indices = []
    test_indices = []
    for combo_idx in range(len(unique_combos)):
        idx_in_combo = np.where(inverse_indices == combo_idx)[0]
        n_obs = len(idx_in_combo)
        n_train = max(1, int(round(n_obs * 0.9)))
        shuffled_idx = np.random.permutation(idx_in_combo)
        train_indices.extend(shuffled_idx[:n_train])
        test_indices.extend(shuffled_idx[n_train:])
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    print(f"Total trials in MAT file: {gtau.shape[2]}")
    print(f"Unique combinations: {len(unique_combos)}")
    print(f"Train trials count: {len(train_indices)}, Test trials count: {len(test_indices)}")
    if sds_channel == '1cm':
        channels = [0]
    elif sds_channel == '2cm':
        channels = [1]
    elif sds_channel == '3cm':
        channels = [2]
    elif sds_channel == 'all':
        channels = [0, 1, 2]
    else:
        raise ValueError(f"Unknown channel type: {sds_channel}")
    def prepare_xy(indices):
        X_out = []
        y_out = []
        for idx in indices:
            for ch in channels:
                curve = gtau[ch, :, idx]
                mean_val = np.mean(curve)
                std_val = np.std(curve)
                if std_val > 0:
                    norm_curve = (curve - mean_val) / std_val
                else:
                    norm_curve = curve - mean_val
                seg1 = norm_curve[0:167]
                seg2 = norm_curve[167:334]
                seg3 = norm_curve[334:501]
                X_sample = np.stack([seg1, seg2, seg3], axis=0)
                X_out.append(X_sample)
                y_sample = [measured_s[idx], measured_d[idx]]
                y_out.append(y_sample)
        return np.array(X_out), np.array(y_out)
    X_train, y_train = prepare_xy(train_indices)
    X_test, y_test = prepare_xy(test_indices)
    print(f"Dataset generated for sds_channel '{sds_channel}':")
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_train, y_train, X_test, y_test

def train_model(mat_path, sds_channel='all', epochs=200, lr=1e-4, batch_size=501, save_path='code/lstm_model.pth'):
    X_train, y_train, X_test, y_test = preprocess_and_split(mat_path, sds_channel)
    train_dataset = AutocorrelationDataset(X_train, y_train)
    test_dataset = AutocorrelationDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    model = DCSLSTMModel(input_size=167, hidden_size=512, num_layers=2, output_size=2, dropout=0.1)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_x.size(0)
        test_loss /= len(test_loader.dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(f"Epoch {epoch+1:03d}/{epochs:03d} | Train MSE: {train_loss:.5f} | Test MSE: {test_loss:.5f}")
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), save_path)
    print(f"Training complete. Best model saved to {save_path}")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions = model(X_test_tensor).cpu().numpy()
    rmse_s = np.sqrt(mean_squared_error(y_test[:, 0], predictions[:, 0]))
    rmse_d = np.sqrt(mean_squared_error(y_test[:, 1], predictions[:, 1]))
    mae_s = np.mean(np.abs(y_test[:, 0] - predictions[:, 0]))
    mae_d = np.mean(np.abs(y_test[:, 1] - predictions[:, 1]))
    r2_s = r2_score(y_test[:, 0], predictions[:, 0])
    r2_d = r2_score(y_test[:, 1], predictions[:, 1])
    print("\n================== Evaluation Results ==================")
    print(f"Shallow Layer flow rate prediction (RMSE): {rmse_s:.4f} ml/min")
    print(f"Shallow Layer flow rate prediction (MAE) : {mae_s:.4f} ml/min")
    print(f"Shallow Layer flow rate prediction (R2)  : {r2_s:.4f}")
    print("--------------------------------------------------------")
    print(f"Deep Layer flow rate prediction (RMSE)   : {rmse_d:.4f} ml/min")
    print(f"Deep Layer flow rate prediction (MAE)    : {mae_d:.4f} ml/min")
    print(f"Deep Layer flow rate prediction (R2)     : {r2_d:.4f}")
    print("========================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train LSTM model for DCS flow rate prediction")
    parser.add_argument('--dataset', type=str, default='code/Dataset_simulated-intra40.mat', help="Path to MAT dataset")
    parser.add_argument('--channel', type=str, default='all', choices=['all', '1cm', '2cm', '3cm'], help="SDS channel configuration")
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--batch-size', type=int, default=501, help="Batch size")
    parser.add_argument('--output-model', type=str, default='code/lstm_model.pth', help="Path to save trained PyTorch model weights")
    args = parser.parse_args()
    train_model(
        mat_path=args.dataset,
        sds_channel=args.channel,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        save_path=args.output_model
    )
