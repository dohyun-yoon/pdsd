import torch
import torch.nn as nn
from nnAudio.features.gammatone import Gammatonegram
from torchmetrics.functional.audio import signal_noise_ratio, signal_distortion_ratio
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility as stoi

class MarginLoss(nn.Module):
    def __init__(self, device):
        super(MarginLoss, self).__init__()
        self.ggram_layer = Gammatonegram(sr=24000, n_fft=1024, hop_length=256, verbose=False).to(device)

    def forward(self, x, m):
        x_ggram = self.ggram_layer(x)
        m_ggram = self.ggram_layer(m)

        diff_ggram = (x_ggram - m_ggram).abs()
        
        return diff_ggram.mean()


class GDLoss(nn.Module):
    def __init__(self, device):
        super(GDLoss, self).__init__()
        self.ggram_layer = Gammatonegram(sr=24000, n_fft=1024, hop_length=256, verbose=False).to(device)

    def gammatonegram_distortion(self, x, y):
        log_diff = 10 * (torch.log10(x) - torch.log10(y))
        mse_per_frame = torch.mean(log_diff**2, dim=0)
        rmse_per_frame = torch.sqrt(mse_per_frame)
        distortion = torch.mean(rmse_per_frame)

        return distortion

    def forward(self, x, m):
        x_ggram = self.ggram_layer(x).squeeze(0)
        m_ggram = self.ggram_layer(m).squeeze(0)
        gd = self.gammatonegram_distortion(x_ggram, m_ggram)

        return gd


class tGDLoss(nn.Module):
    def __init__(self, device):
        super(tGDLoss, self).__init__()
        self.ggram_layer = Gammatonegram(sr=24000, n_fft=1024, hop_length=256, verbose=False).to(device)

    def gammatonegram_distortion(self, x, y):
        x_ggram = self.ggram_layer(x).squeeze(0)
        y_ggram = self.ggram_layer(y).squeeze(0)
        log_diff = 10 * (torch.log10(x_ggram) - torch.log10(y_ggram))
        mse_per_frame = torch.mean(log_diff**2, dim=0)
        rmse_per_frame = torch.sqrt(mse_per_frame)
        distortion = torch.mean(rmse_per_frame)

        return distortion

    def forward(self, x, y, targetGD):
        gd = self.gammatonegram_distortion(x, y)
        loss = (gd - targetGD).abs()

        return loss


class SNRLoss(nn.Module):
    def __init__(self):
        super(SNRLoss, self).__init__()
        
    def getSNR(self, x, m):
        snr = signal_noise_ratio(m, x)
        
        return snr.squeeze().item()
        
    def forward(self, x, m, targetSNR):
        snr = signal_noise_ratio(m, x)
        loss = (snr - targetSNR).abs()
        
        return loss.mean()


class SDRLoss(nn.Module):
    def __init__(self):
        super(SDRLoss, self).__init__()
        
    def getSDR(self, x, m):
        sdr = signal_distortion_ratio(m, x)
        
        return sdr.squeeze().item()
        
    def forward(self, x, m, targetSDR):
        snr = signal_distortion_ratio(m, x)
        loss = (snr - targetSDR).abs()
        
        return loss.mean()


class STOILoss(nn.Module):
    def __init__(self):
        super(STOILoss, self).__init__()
        
    def getSTOI(self, x, y):
        return stoi(y.cpu(), x.cpu(), 24000).squeeze().item()
    
    def forward(self, x, y, targetSTOI):
        device = x.device
        predSTOI = stoi(y.cpu(), x.cpu(), 24000).to(device)
        loss = (predSTOI - targetSTOI).abs()
        
        return loss


class NoiseAdder(nn.Module):
    def __init__(self, x_len):
        super(NoiseAdder, self).__init__()
        self.noise = nn.Parameter(torch.randn(1, x_len))
        
    def forward(self, x):
        return x + self.noise