import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import soundfile as sf
from tqdm import tqdm

from utils import *


def train(wav_dir, tSNR, tGD):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    x, sr = torchaudio.load(wav_dir)
    x = x.unsqueeze(0).to(device)
    
    model = NoiseAdder(x.shape[-1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    recloss = tGDLoss(device)
    simloss = nn.CosineSimilarity(dim=-1)
    snrloss = SNRLoss()
    
    patience = 10
    best_loss = float('inf')
    counter = 0
    
    model.train()
    for _ in range(10000 if tSNR != 45 else 20000):
        optimizer.zero_grad()
        
        y = model(x)
        
        loss_P = recloss(x, y, torch.tensor([tGD]).type(x.dtype).unsqueeze(0).to(device))
        loss_S = 1 - simloss(x, y).mean()
        loss_T = snrloss(x, y, torch.tensor([tSNR]).type(x.dtype).unsqueeze(0).to(device))
        
        if torch.isnan(loss_P) or torch.isnan(loss_S) or torch.isnan(loss_T):
            break
        loss = loss_P + loss_S + loss_T
        
        loss.backward()
        optimizer.step()
        
        # early stopping
        if loss < best_loss:
            if loss < best_loss:
                best_loss = loss
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            break
    
    y_last = model(x).detach()
    pSNR = snrloss.getSNR(x, y_last)
    pGD = recloss.gammatonegram_distortion(x, y_last)
    if abs(tSNR-pSNR) < 1.0 and abs(tGD-pGD) < 1.0:
        return y_last.squeeze().cpu(), sr, pSNR, pGD
    else:
        return None, None, None, None


if __name__ == '__main__':
    combs = [
        [5, 6.25], [5, 8.75], [5, 11.25], [5, 13.75], [5, 16.25], 
        [15, 3.75], [15, 6.25], [15, 8.75], [15, 11.25], 
        [25, 1.25], [25, 3.75], [25, 6.25], 
        [35, 1.25], [45, 1.25], 
    ]
    
    wavs_dirs = glob.glob('./*.wav')
    for wav_dir in tqdm(wavs_dirs):
        utt_name = os.path.basename(wav_dir)[:-4]
        os.makedirs(f'./results/{utt_name}', exist_ok=True)
        cnt = [0] * 14
        for i, (tSNR, tGD) in tqdm(enumerate(combs)):
            for _ in range(50):
                y, sr, pSNR, pGD = train(wav_dir, tSNR, tGD)
                if y is not None:
                    sf.write(f'./results/{utt_name}/SNR_{pSNR:.2f}-GD_{pGD:.2f}.wav', y, sr)
                    cnt[i] = 1
        if all(x==1 for x in cnt): break
                
    print('all is well')