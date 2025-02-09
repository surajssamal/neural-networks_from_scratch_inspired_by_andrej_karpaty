#!/bin/python3
import torch
import torch.nn.modules as nn
import torch.optim as optim
import torch.nn.functional as F
from Fetch_stock_data import stock_batches

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--stock", type=str, required=True)
args = parser.parse_args()
Company = args.stock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Scaling:
    def __init__(self,data=list):
        self.data = data
        self.min_value=min(data)
        self.max_value = max(data)
    def Normalization(self,denormalize=None,normalization=None):
        if denormalize ==None and normalization== None:
            normalized_data =[(i - self.min_value)/(self.max_value-self.min_value) for i in self.data]
            return normalized_data
        elif normalization != None:
            if isinstance(denormalize, (float, int)):
                return (normalization - self.min_value)/(self.max_value-self.min_value)
            normalized_data =[(i - self.min_value)/(self.max_value-self.min_value) for i in normalization]
            return normalized_data
    
        else:
    
            if isinstance(denormalize, (float, int)):
                return denormalize * (self.max_value - self.min_value) +self.min_value 
            denormalize_data = [i * (self.max_value - self.min_value) + self.min_value for i in denormalize]
            return denormalize_data
            
class CustomLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # LSTM gates
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.cell_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Final output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Activations
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden_state=None):

        
        batch_size, seq_len, input_dim = x.size()
        h_t, c_t = (torch.zeros(batch_size, self.hidden_dim, device=x.device),
                    torch.zeros(batch_size, self.hidden_dim, device=x.device)) if hidden_state is None else hidden_state

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat((x_t, h_t), dim=1)

            # Gates
            f_t = self.sigmoid(self.forget_gate(combined))
            i_t = self.sigmoid(self.input_gate(combined))
            g_t = self.tanh(self.cell_gate(combined))
            o_t = self.sigmoid(self.output_gate(combined))

            # Update cell state and hidden state
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * self.tanh(c_t)

            outputs.append(h_t)

        outputs = torch.stack(outputs, dim=1)
        outputs = self.fc(outputs)
        return outputs
def traning_model(model,data):
    x,y = data
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epochs = 5000
    running_loss = 0
    for i in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs= model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 ==0 :
            print(f"running loss : {running_loss} and loss : {loss}")
            running_loss = 0
batch_size =5
seq_len =3
data = stock_batches.Fetch_data(f"{Company}",80)
c = Scaling(data)
normalized_data = c.Normalization()
x_train,y_train,rest = stock_batches.get_batch(batch_size=5,seq_len=3,split=normalized_data)

loss_fn = nn.L1Loss()
model = CustomLSTM(3,32,1)
traning_model(model,data=(x_train,y_train))

todays_prediction =torch.tensor(c.Normalization(normalization=data[-seq_len:]))
prediction = c.Normalization(denormalize=model(todays_prediction.view(1,1,-1)).item())
prediction
print(prediction)

