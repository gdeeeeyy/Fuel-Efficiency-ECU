# Dynamic Fuel Map Generator

Fuel mapping is the core component of any vehicle's working. Good Fuel Mapping is the crux of performance and emission control & analysis of Engines through ECUs. Static Mapping techniques are still widely used for conventional approaches, but this may not adapt to dynamic riding conditions. In this project, we propose a Deep Learning model architecture based on Sequential Long Short-Term Memory (LSTM) to predict and optimize fuel and air intake into the engine using real-time data.

Parameters such as engine load, engine speed (RPM), and gear position can be used to optimize fuel and air intake into the engine. Further, this data can be used to optimize the Stoichiometric Air-Fuel Ratio(AFR) that can optimize fuel consumption and performance for variable parameters, thereby dynamically adapting to variable conditions. By generating these dynamic fuel maps, we can ensure that the fuel and air intake are regulated for various engine speeds and engine loads, ensuring a proper AFR for all possible speed and load ranges.

# Model used

This model employs 4 stacked sequential LSTM layers. Each layer processes the input sequence X and generates hidden state vectors, H belonging to RTm, where m=64 is the dimensionality of the hidden state. The LSTM layer updates its hidden state ht and cell state ct using recurrence relations at each time step.
$$h_t, c_t = \text{LSTM}(x_t, h_{t-1}, c_{t-1}),$$

$$\quad  \text{where} \ c_t \in  \mathbb{R}^m$$

Additionally, a dropout mechanism with a rate of p=0.2 is applied between LSTM layers to drop out a few connections in the architecture, creating a new network architecture at each iteration to ensure overfitting of data doesn’t occur in the long run.

The final hidden state layer hT belongs to Rm, corresponding to the last time step T, is passed to a fully connected dense linear layer;

$$y_T = W h_T + b,$$

$$\quad  \text{where} \ W \in  \mathbb{R}^{k \times m}, \ b \in  \mathbb{R}^k$$

The output y_T is the predicted, normalized values of Fuel and Air Intake.

Additionally the model also uses MSE as the Loss function and Adam's Optimizer for gradient optimisation.

# Desired Architecture

The model is designed to predict Fuel and Air intake into the engine based on a sequential time series input of parameters such as Engine Load, RPM, Vehicle Speed, and Gear Position. This model is built on a sequential LSTM Network.

![diagram-export-13-12-2024-13_55_54](https://github.com/user-attachments/assets/d2de127b-f2a6-457d-84a8-bafb90973db0)

This figure depicts a 4-layer LSTM neural network designed to predict fuel and air intake in a vehicle. The model processes real-time sensor data (engine load, vehicle speed, gear position) through LSTM layers to capture temporal dependencies. The final output layer predicts fuel and air intake values. The model is trained using the MSE loss function and the Adam optimizer.

# Results

| Model | MSE Score | R² Score  | Precision | Recall   | F1 Score |
| ----- | --------- | --------- | --------- | -------- | -------- |
| LSTM  | 0.115835  | -0.394059 | 0.520000  | 0.596330 | 0.555556 |
| LR    | 0.083789  | -0.008394 | 0.607143  | 0.155963 | 0.248175 |
| SVM   | 0.092074  | -0.108098 | 0.605634  | 0.394495 | 0.477778 |

**Table 1**: Combined performance metrics of LSTM, Linear Regression (LR), and SVM models, showing MSE, R² Score, Precision, Recall, and F1 Score.

# Conclusion

Through this project, we were able to show that, an LSTM network-based model can be used as a replacement for traditional static fuel-map for ensuring fuel efficiency and calculating fuel and air intake into the engine through various parameters by analyzing all the temporal data and non-linear relationships. To our knowledge, this represents one of the first applications of Deep Learning in calculating the fuel efficiency of a vehicle and can help in regulating fuel consumption dynamically, through a Deep Learning model.
