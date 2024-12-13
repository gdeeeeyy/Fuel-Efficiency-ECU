import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# Define the LSTM Model for 4 Input Features
class FuelAirPredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(FuelAirPredictionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x): #processes input through LSTM
        lstmOut, _ = self.lstm(x)
        lastOut = lstmOut[:, -1, :]
        output = self.fc(lastOut)
        return output

# Prepares the dataset into a time series -> this function splits it into sliding windows of "seqLen"
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seqLen):
        self.features = data[:, :-2]  # Input features: Engine_Load, RPM, Vehicle_Speed, Gear_Position
        self.targets = data[:, -2:]  # Targets: Fuel_Intake, Air_Intake
        self.seqLen = seqLen

    def __len__(self):
        return len(self.features) - self.seqLen + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seqLen]
        y = self.targets[idx + self.seqLen - 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def loadData(filePath, seqLen):
    """
    Load and preprocess the data, aligning it with the required structure.
    """
    df = pd.read_csv(filePath)
    df = df.rename(columns={
        'Engine_Load_Pressure(%)': 'Engine_Load',
        'Vehicle_Speed(km/h)': 'Vehicle_Speed',
        'O2_Intake(g/s)': 'Air_Intake',
        'Fuel_Amount(mL/s)': 'Fuel_Intake',
        'Gear_Position': 'Gear_Position'
    })
    
    # Separate input features and targets
    ipFeatures = df[['Engine_Load', 'RPM', 'Vehicle_Speed', 'Gear_Position']]
    tgFeatures = df[['Fuel_Intake', 'Air_Intake']]
    
    # Fit separate scalers
    ipScaler = MinMaxScaler()
    tgScaler = MinMaxScaler()
    
    scaledIps = ipScaler.fit_transform(ipFeatures.values)
    scaledTgts = tgScaler.fit_transform(tgFeatures.values)
    
    # Combine scaled inputs and targets
    scaledData = np.hstack((scaledIps, scaledTgts))
    
    # Create dataset with sequence length
    dataset = TimeSeriesDataset(scaledData, seqLen)
    
    return dataset, ipScaler, tgScaler

# Function to calculate AFR
def calcAfr(fuelIntake, airIntake):
    return airIntake / fuelIntake

def predictAndOpt(model, newData, seqLen, ipScaler, tgScaler):
    """
    Predict fuel and air intake, then calculate and optimize AFR.
    """
    model.eval()
    
    # Scale new_data using input_scaler
    scaledData = ipScaler.transform(newData)  # Only scale the 4 input features
    inputs = torch.tensor(scaledData, dtype=torch.float32).unsqueeze(0)  # Add batch and sequence dimension
    
    # Model prediction
    with torch.no_grad():
        predictions = model(inputs).numpy()  # Shape: (1, 2)
    
    # Inverse transform predictions using targetScaler
    predictions = tgScaler.inverse_transform(predictions)
    
    # Extract predictions
    fuelIntake, airIntake = predictions[0, 0], predictions[0, 1]
    afr = calcAfr(fuelIntake, airIntake)

    # Optimize AFR
    if afr > 15.0:  # Too lean
        airIntake *= 0.95
    elif afr < 13.0:  # Too rich
        airIntake *= 1.05

    optimizedAfr = calcAfr(fuelIntake, airIntake)
    return {
        "Fuel Intake": fuelIntake,
        "Air Intake": airIntake,
        "Optimized AFR": optimizedAfr
    }

# Main function
if __name__ == "__main__":
    # Hyperparameters
    ipSize = 4  # Number of input features (Engine_Load, RPM, Vehicle_Speed, Gear_Position)
    hiddenSize = 64
    numLayers = 4
    opSize = 2  # Predicting fuel and air intake
    seqLen = 10
    batchSize = 32
    learningRate = 0.001
    epochs = 1000

    # Load dataset
    filePath = "vehicleData.csv"  # Replace with your file path
    dataset, ipScaler, tgScaler = loadData(filePath, seqLen)
    trainSize = int(len(dataset) * 0.8)
    testSize = len(dataset) - trainSize
    trainDataset, testDataset = torch.utils.data.random_split(dataset, [trainSize, testSize])

    # DataLoaders
    train_loader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    test_loader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)

    # Initialize model, loss, and optimizer
    model = FuelAirPredictionLSTM(ipSize, hiddenSize, numLayers, opSize, dropout=0.2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        test_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Test the model on validation set
        model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "fuelAirPredictionLSTM.pth")

    # Load the model for predictions
    model.load_state_dict(torch.load("fuelAirPredictionLSTM.pth"))
    model.eval()

    # Generate prediction for a sample input
    sampleInput = np.array([[75, 2500, 80, 3]])  # Replace with your own sample input data
    output = predictAndOpt(model, sampleInput, seqLen, ipScaler, tgScaler)

    # Display output
    print(f"Predicted Fuel Intake: {output['Fuel Intake']:.4f} mL/s")
    print(f"Predicted Air Intake: {output['Air Intake']:.4f} g/s")
    print(f"Optimized AFR: {output['Optimized AFR']:.4f}")
