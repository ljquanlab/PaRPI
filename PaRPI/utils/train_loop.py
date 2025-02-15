from __future__ import print_function
from tqdm import tqdm
import numpy as np
import torch
import utils.metrics as metrics



def train(model, device, train_loader, criterion, optimizer, epoch, batch_size):
    model.train()
    met = metrics.MLMetrics(objective='binary')

    for batch_idx, (x, x0, x00, x000, y0) in enumerate(train_loader):
        g, b, s, p, y = x.to(device), x0.float().to(device), x00.float().to(device), x000.float().to(device), y0.to(device).float()
        batch_e = x.edata['feat'].to(device)

        if y0.sum() == 0 or y0.sum() == batch_size:
            continue
        optimizer.zero_grad()  
        output = model(g, batch_e, b, s, p)
        loss = criterion(output, y)
        prob = torch.sigmoid(output)
        y_np = y.to(device='cpu', dtype=torch.long).detach().numpy()
        p_np = prob.to(device='cpu').detach().numpy()
        met.update(y_np, p_np, [loss.item()])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
    return met



def validate(model, device, test_loader, criterion):
    model.eval()
    y_all = []
    p_all = []
    l_all = []
    with torch.no_grad():
        for batch_idx, (x, x0, x00, x000, y0) in enumerate(test_loader):
            g, b, s, p, y = x.to(device), x0.float().to(device), x00.float().to(device), x000.float().to(device), y0.to(device).float()
            batch_e = x.edata['feat'].to(device)

            output = model(g, batch_e, b, s, p)
            loss = criterion(output, y)
            prob = torch.sigmoid(output)  # 将输出控制到[0,1]
            y_np = y.to(device='cpu', dtype=torch.long).numpy()
            p_np = prob.to(device='cpu').numpy()
            l_np = loss.item()

            y_all.append(y_np)
            p_all.append(p_np)
            l_all.append(l_np)

    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    l_all = np.array(l_all)

    met = metrics.MLMetrics(objective='binary')
    met.update(y_all, p_all, [l_all.mean()])  

    return met, y_all, p_all

