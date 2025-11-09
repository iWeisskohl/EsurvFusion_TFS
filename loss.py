import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans
from torch.distributions.normal import Normal
import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
import matplotlib.pyplot as plt

class Loss_function(nn.Module):
    def __init__(self):
        super(Loss_function, self).__init__()

    def forward(self, y, nu, events, xi, rho, pred, lambd=None):
        n = len(y)
        num_sources = len(pred['mux'])
        total_loss = torch.zeros((num_sources), dtype=torch.float64)

        for i in range(num_sources):
            mux = pred["mux"][i]
            sig2x = pred["sig2x"][i]
            hx = pred["hx"][i]
            penalty1 = pred["penalty1"][i]
            penalty2 = pred["penalty2"][i]

            sigx = torch.sqrt(sig2x)
            Z2 = hx * sig2x + 1
            Z = torch.sqrt(Z2)
            sig1 = sigx * Z

            pl = 1 / Z * torch.exp(-0.5 * hx * (y - mux) ** 2 / Z2)

            # Bel
            eps = 1e-4 * torch.std(y)
            norm_dist = Normal(mux, sigx)
            Sy1 = 1 - norm_dist.cdf(y) - pl + pl * Normal(mux, sig1).cdf(y)
            

            pl1 = 1 / Z * torch.exp(-0.5 * hx * (y - eps - mux) ** 2 / Z2)
            pl2 = 1 / Z * torch.exp(-0.5 * hx * (y + eps - mux) ** 2 / Z2)

            # Pl
            Sy2 = 1 - norm_dist.cdf(y) + pl * Normal(mux, sig1).cdf(y)
            Fy2_1 = norm_dist.cdf(y + eps) + pl1 * Normal(mux, sig1).cdf(y - eps)
            Fy2_2 = norm_dist.cdf(y - eps) - pl2 * (1 - Normal(mux, sig1).cdf(y + eps))

            fy2 = Fy2_1 - Fy2_2
            fy1 = fy2 - pl1 * Normal(mux, sig1).cdf(y) - pl2 * (1 - Normal(mux, sig1).cdf(y))

            Sy1 = torch.clamp(Sy1, min=0.0)
            Sy2 = torch.clamp(Sy2, min=0.0)

            fy1 = torch.clamp(fy1, min=0.0)
            fy2 = torch.clamp(fy2, min=0.0)

            loss = -lambd * torch.mean(torch.log(fy1 + nu) * events + torch.log(Sy1 + nu) * (1 - events)) \
                   - (1 - lambd) * torch.mean(torch.log(fy2 + nu) * events + torch.log(Sy2 + nu) * (1 - events)) \
                   + xi * penalty1 + rho * penalty2
            
            total_loss[i] = loss
        total_loss[-1] *= 0.01
        return torch.mean(total_loss)
        
