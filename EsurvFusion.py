import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Survival_prediction(nn.Module):
    def __init__(self, X_dict, K_dict, prototypes, trainable_lambda=True):
        super(Survival_prediction, self).__init__()
        self.X_dict = X_dict
        self.K_dict = K_dict
        self.prototypes = prototypes

        self.input_dim_list = [X.shape[1] for X in X_dict.values()]
        self.prototype_dim_list = [K for K in K_dict.values()]
        self.num_sources = len(self.input_dim_list)

        self.alphas = nn.ParameterList([Parameter(torch.Tensor(1, k)) for k in self.prototype_dim_list])
        self.betas = nn.ParameterList([Parameter(torch.Tensor(k, p)) for k, p in zip(self.prototype_dim_list, self.input_dim_list)])
        self.sigs = nn.ParameterList([Parameter(torch.Tensor(1, k)) for k in self.prototype_dim_list])
        self.etas = nn.ParameterList([Parameter(torch.Tensor(1, k)) for k in self.prototype_dim_list])
        self.gammas = nn.ParameterList([Parameter(torch.Tensor(k, 1)) for k in self.prototype_dim_list])
        self.ws = nn.ParameterList([Parameter(torch.Tensor(k, p)) for k, p in zip(self.prototype_dim_list, self.input_dim_list)])
        self.discounts = nn.ParameterList([Parameter(torch.tensor([3.0])) for i in range(self.num_sources)])
        if trainable_lambda:
            self.z = nn.ParameterList([Parameter(torch.tensor([0.0])) for i in range(self.num_sources)])

        self.reset_parameters(prototypes)
    def reset_parameters(self, prototypes):
        for i, prototype in enumerate(prototypes):
            self.alphas[i] = Parameter(prototype['alpha'])
            self.betas[i] = Parameter(prototype['Beta'])
            self.sigs[i] = Parameter(prototype['sig'])
            self.etas[i] = Parameter(prototype['eta'])
            self.gammas[i] = Parameter(prototype['gam'])
            self.ws[i] = Parameter(prototype['W'])

    def forward(self, inputs):
        assert isinstance(inputs, dict) and all(torch.is_tensor(input) for input in inputs.values())
        nt = next(iter(inputs.values())).size(0)

        mux_total = torch.zeros((self.num_sources, nt), dtype=torch.float64)
        sig2x_total = torch.zeros((self.num_sources, nt), dtype=torch.float64)
        hx_total = torch.zeros((self.num_sources, nt), dtype=torch.float64)
        hx_dc = torch.zeros((self.num_sources, nt), dtype=torch.float64)

        for i, (key, input) in enumerate(inputs.items()):
            h = self.etas[i] ** 2

            a = torch.zeros(nt, self.prototype_dim_list[i])
            for k in range(self.prototype_dim_list[i]):
                a[:, k] = torch.exp(
                    -self.gammas[i][k] ** 2 * torch.sum((input - self.ws[i][k, :].unsqueeze(0).expand(nt, -1)) ** 2, dim=1))

            H = h.expand(nt, -1)
            hx = torch.sum(a * H, dim=1)
            mu = torch.mm(input, self.betas[i].T) + self.alphas[i].expand(nt, -1)
            mux = torch.sum(mu * a * H, dim=1) / hx
            sig2x = torch.sum((self.sigs[i] ** 2).expand(nt, -1) * (a ** 2) * (H ** 2), dim=1) / (hx ** 2)

            mux_total[i] =  mux
            sig2x_total[i] = sig2x
            
            # possibilistic discounting
            hx_total[i] =  hx
            hx_dc[i] =  hx*torch.sigmoid(self.discounts[i])


        mux_comb = torch.sum(mux_total*hx_dc, dim=0)/torch.sum(hx_dc, dim=0)
        sig2x_comb = torch.sum(sig2x_total*hx_dc**2, dim=0)/torch.sum(hx_dc, dim=0)**2
        hx_comb = torch.sum(hx_dc, dim=0)

        mux_final = torch.cat((mux_total, mux_comb.unsqueeze(0)), dim=0)
        sig2x_final = torch.cat((sig2x_total, sig2x_comb.unsqueeze(0)), dim=0)
        hx_final = torch.cat((hx_total, hx_comb.unsqueeze(0)), dim=0)

        return {"mux": mux_final, "sig2x": sig2x_final, "hx": hx_final}



