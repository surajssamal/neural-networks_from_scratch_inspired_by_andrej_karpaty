#!/bin/python3 
import torch
import datetime as dt
import yfinance as yf
class stock_batches:
    @staticmethod
    def get_batch(batch_size,seq_len,split=None):
        data = torch.tensor(split)
        ix = range(0,len(data)-seq_len)
        x = torch.stack([data[i:i+seq_len] for i in ix])
        y = torch.stack([data[i+1:1+i+seq_len][-1] for i in ix])
        no_of_batch = int(len(x)/batch_size)
        x_batch = x[:no_of_batch*batch_size]
        y_batch = y[:no_of_batch*batch_size].view(-1,batch_size)
        x_batch = x_batch.view(no_of_batch,batch_size,seq_len)
        
        # Handle the remaining data (if any)
        leftover_x = x[no_of_batch * batch_size:]
        leftover_y = y[no_of_batch * batch_size:]
    
        if len(leftover_x) > 0:
            pad_length = batch_size - len(leftover_x)
            padding_x = torch.zeros((pad_length, seq_len))  # Padding with zeros
            padding_y = torch.zeros((pad_length,))  # Padding with zeros
    
            # Append padding to leftover data
            leftover_x = torch.cat([leftover_x, padding_x], dim=0)
            leftover_y = torch.cat([leftover_y, padding_y], dim=0)
            
            leftovers = (leftover_x,leftover_y)
            print(f"leftovers : {len(leftover_x)/batch_size}")
        
        return x_batch,y_batch.unsqueeze(2),(leftovers if len(leftover_x) else None)

    @staticmethod
    def Fetch_data(stock,no_of_days):
        company =stock
        days_back = dt.timedelta(days=no_of_days)
        end = dt.date.today()
        start = end-days_back
        
        data = yf.download(company, start, end)
        data_formated = data[data.columns[1]].values.tolist()
        return data_formated
