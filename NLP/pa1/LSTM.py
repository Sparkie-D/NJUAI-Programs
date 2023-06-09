import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class MyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, bidirectional=True)

        self.linear = nn.Linear(hidden_layer_size*2, output_size)

        self.hidden_cell = (torch.zeros(2, 1, self.hidden_layer_size), torch.zeros(2, 1, self.hidden_layer_size))
        # self.hidden_cell = torch.zeros(1, 1, self.hidden_layer_size).to(device)
        self.to(device)
        self.loss_func = nn.MSELoss().to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)


    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions.reshape(-1)

    def fit(self, X, Y):
        epochs = 20
        X, Y = torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(Y, dtype=torch.float32).to(device)
        print("Training LSTM model, device=", device)
        for i in range(epochs):
            epoch_loss = 0
            start = 0
            batch_size = 32
            while start < (len(X)):
                inputs = X[start:start+batch_size].to(device)
                targets = Y[start:start+batch_size].to(device)

                self.optimizer.zero_grad()
                self.hidden_cell = (torch.zeros(2, 1, self.hidden_layer_size).to(device), torch.zeros(2, 1, self.hidden_layer_size).to(device))
                y_pred = self.forward(inputs)

                # print(y_pred.shape, targets.shape)
                loss = self.loss_func(y_pred, targets)
                loss.backward()
                self.optimizer.step()

                start += batch_size
                epoch_loss += loss

            print(f'epoch: {i:3} loss: {epoch_loss.item():10.8f}')

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(device)
        pred = self.forward(X)
        bound = 0.5
        return [1 if pred[i] > bound else 0 for i in range(len(pred))]


if __name__ == '__main__':
    epochs = 100
    model = MyLSTM()
    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

