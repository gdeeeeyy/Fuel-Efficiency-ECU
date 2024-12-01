import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# Define the LSTM model
class FuelAirPredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(FuelAirPredictionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Check the dimensions of lstm_out
        if lstm_out.dim() == 3:
            last_out = lstm_out[:, -1, :]  # Select the last output from the sequence
        elif lstm_out.dim() == 2:
            last_out = lstm_out  # If the sequence length is 1, we directly use the output
        else:
            raise ValueError("LSTM output has an unexpected number of dimensions")
        
        output = self.fc(last_out)
        return output

# Define a Time Series Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.features = data[:, :-2]
        self.targets = data[:, -2:]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len - 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Function to load data and preprocess
def load_data(file_path, seq_len):
    df = pd.read_csv(file_path)
    df = df.rename(columns={
        'Engine_Load_Pressure(%)': 'Engine_Load',
        'Vehicle_Speed(km/h)': 'Vehicle_Speed',
        'O2_Intake(g/s)': 'Air_Intake',
        'Fuel_Amount(mL/s)': 'Fuel_Intake',
        'Gear_Position': 'Gear_Position'
    })

    # Scale data
    ip_scaler = MinMaxScaler()
    tg_scaler = MinMaxScaler()
    scaled_ips = ip_scaler.fit_transform(df[['Engine_Load', 'RPM', 'Vehicle_Speed', 'Gear_Position']])
    scaled_tgts = tg_scaler.fit_transform(df[['Fuel_Intake', 'Air_Intake']])

    scaled_data = np.hstack((scaled_ips, scaled_tgts))
    dataset = TimeSeriesDataset(scaled_data, seq_len)
    return dataset, ip_scaler, tg_scaler

# Define a function to evaluate the models (MAE, MSE, R2)
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, r2, mae

# Function to calculate Precision, Recall, and F1 Score (Binarized)
def calculate_precision_recall_f1(y_true, y_pred, threshold=0.5):
    # Apply the threshold to both true values and predicted values
    y_true_bin = (y_true > threshold).astype(int)
    y_pred_bin = (y_pred > threshold).astype(int)
    
    # Calculate precision, recall, and f1 score
    precision = precision_score(y_true_bin, y_pred_bin, average='binary')
    recall = recall_score(y_true_bin, y_pred_bin, average='binary')
    f1 = f1_score(y_true_bin, y_pred_bin, average='binary')
    return precision, recall, f1

# Main function
if __name__ == "__main__":
    seq_len = 10
    batch_size = 32
    hidden_size = 64
    num_layers = 4
    learning_rate = 0.001
    epochs = 1000

    file_path = "vehicleData.csv"  # Replace with your dataset
    dataset, ip_scaler, tg_scaler = load_data(file_path, seq_len)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize and train LSTM
    model = FuelAirPredictionLSTM(input_size=4, hidden_size=hidden_size, num_layers=num_layers, output_size=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Prepare data for evaluation
    X = dataset.features
    y = dataset.targets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    # Train and evaluate SVM
    svm_model = SVR()
    svm_model.fit(X_train, y_train[:, 0])  # Train SVM on Fuel Intake only
    svm_preds = svm_model.predict(X_test)

    # Ensure predictions and true values are of same length
    lr_preds = lr_preds[:len(y_test)]
    svm_preds = svm_preds[:len(y_test)]

    # Evaluate LSTM predictions
    lstm_preds = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()

    # Check if lstm_preds is 2D and slice accordingly
    if len(lstm_preds.shape) > 1:
        lstm_preds = lstm_preds[:, 0]  # Select only the first column (Fuel Intake)

    lstm_preds = lstm_preds[:len(y_test)]  # Ensure same length

    # Calculate metrics
    lstm_metrics = evaluate_model(y_test[:, 0], lstm_preds)  # Only consider Fuel Intake
    lr_metrics = evaluate_model(y_test[:, 0], lr_preds[:, 0])
    svm_metrics = evaluate_model(y_test[:, 0], svm_preds)

    # Calculate Precision, Recall, and F1 Score for each model
    lr_precision, lr_recall, lr_f1 = calculate_precision_recall_f1(y_test[:, 0], lr_preds[:, 0], threshold=0.5)
    svm_precision, svm_recall, svm_f1 = calculate_precision_recall_f1(y_test[:, 0], svm_preds, threshold=0.5)
    lstm_precision, lstm_recall, lstm_f1 = calculate_precision_recall_f1(y_test[:, 0], lstm_preds, threshold=0.5)

    # Comparison Table
    comparison_df = pd.DataFrame({
        "Model": ["LSTM", "Linear Regression", "SVM"],
        "MSE Score": [lstm_metrics[0], lr_metrics[0], svm_metrics[0]],
        "R² Score": [lstm_metrics[1], lr_metrics[1], svm_metrics[1]],
        "MAE Score": [lstm_metrics[2], lr_metrics[2], svm_metrics[2]],
        "Precision": [lstm_precision, lr_precision, svm_precision],
        "Recall": [lstm_recall, lr_recall, svm_recall],
        "F1 Score": [lstm_f1, lr_f1, svm_f1]
    })

    # Display the raw comparison table
    print("Comparison Table (Metrics in raw values):")
    print(comparison_df)
