import torch
import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv

def Evaluation(pred, durations_test, events_test, weight, pt=None, YJ=False):

    mux = pred['mux'][-1]
    sigx = torch.sqrt(pred['sig2x'][-1])
    hx = pred['hx'][-1]
    Z2 = hx * pred['sig2x'][-1] + 1
    Z = torch.sqrt(Z2)
    sig1 = sigx * Z

    time_grid = np.linspace(durations_test.numpy().min(), durations_test.numpy().max(), 100)
    if YJ == False:
        D, M = torch.meshgrid(torch.log(durations_test), mux)
        diff = D - M

        pl = 1 / Z * torch.exp(-0.5 * hx * diff ** 2 / Z2)
        Fy1 = torch.distributions.Normal(mux, sigx).cdf(D) - pl * torch.distributions.Normal(mux, sig1).cdf(D)
        Fy2 = Fy1 + pl

        surv_df = 1 - (weight * Fy1 + (1 - weight) * Fy2)
        surv_df = pd.DataFrame(surv_df.detach().numpy(), index=durations_test.numpy())

        ev = EvalSurv(surv_df, durations_test.numpy(), events_test, censor_surv='km')

        c_index = ev.concordance_td('adj_antolini')
        _ = ev.brier_score(time_grid).plot()
        IBS = ev.integrated_brier_score(time_grid)
        NBLL = ev.integrated_nbll(time_grid)
    else:
        D, M = torch.meshgrid(YJtransform(durations_test, pt), mux)
        diff = D - M

        pl = 1 / Z * torch.exp(-0.5 * hx * diff ** 2 / Z2)
        Fy1 = torch.distributions.Normal(mux, sigx).cdf(D) - pl * torch.distributions.Normal(mux, sig1).cdf(D)
        Fy2 = Fy1 + pl

        surv_df = 1 - (weight * Fy1 + (1 - weight) * Fy2)
        surv_df = pd.DataFrame(surv_df.detach().numpy(), index=durations_test.numpy())

        ev = EvalSurv(surv_df, durations_test.numpy(), events_test, censor_surv='km')

        c_index = ev.concordance_td('adj_antolini')
        _ = ev.brier_score(time_grid).plot()
        IBS = ev.integrated_brier_score(time_grid)
        NBLL = ev.integrated_nbll(time_grid)

    return c_index, IBS, NBLL

def YJtransform(x, pt):
    return torch.tensor(pt.transform(torch.log(x.reshape(-1, 1))).squeeze(), dtype=torch.float64)

