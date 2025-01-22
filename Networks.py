import torch
import torch.nn as nn
from functools import reduce
import operator
import torch.nn as nn
import torch.nn.functional as F
from utils import View

def dense_nn(
    num_features,
    num_targets,
    num_layers,
    intermediate_size=10,
    activation='relu',
    output_activation='sigmoid',
):
    if num_layers > 1:
        if intermediate_size is None:
            intermediate_size = max(num_features, num_targets)
        if activation == 'relu':
            activation_fn = torch.nn.ReLU
        elif activation == 'sigmoid':
            activation_fn = torch.nn.Sigmoid
        else:
            raise Exception('Invalid activation function: ' + str(activation))
        net_layers = [torch.nn.Linear(num_features, intermediate_size), activation_fn()]
        for _ in range(num_layers - 2):
            net_layers.append(torch.nn.Linear(intermediate_size, intermediate_size))
            net_layers.append(activation_fn())
        if not isinstance(num_targets, tuple):
            net_layers.append(torch.nn.Linear(intermediate_size, num_targets))
        else:
            net_layers.append(torch.nn.Linear(intermediate_size, reduce(operator.mul, num_targets, 1)))
            net_layers.append(View(num_targets))
    else:
        if not isinstance(num_targets, tuple):
            net_layers = [torch.nn.Linear(num_features, num_targets)]
        else:
            net_layers = [torch.nn.Linear(num_features, reduce(operator.mul, num_targets, 1)), View(num_targets)]

    if output_activation == 'relu':
        net_layers.append(torch.nn.ReLU())
    elif output_activation == 'sigmoid':
        net_layers.append(torch.nn.Sigmoid())
    elif output_activation == 'tanh':
        net_layers.append(torch.nn.Tanh())
    elif output_activation == 'softmax':
        net_layers.append(torch.nn.Softmax(dim=-1))

    return torch.nn.Sequential(*net_layers)

class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        return torch.clamp(x, min=-1)



class dense2_nn(nn.Module):
    def __init__(self, num_features,
    num_targets,
    intermediate_size=10,
):
        super(dense2_nn, self).__init__()
        self.linear1 = nn.Linear(num_features, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, intermediate_size)
        self.linear3 = nn.Linear(intermediate_size, int(num_targets/2))
        self.linear4 = nn.Linear(intermediate_size, int(num_targets/2))
        self.act = nn.Softplus()
        self.cusact = CustomActivation()


    def forward(self, x):
        x = x.float()
        out = F.relu(self.linear1(x)) # activation
        out = F.relu(self.linear2(out)) # activation
        out1 = self.cusact(self.linear3(out)) # activation
        out2 = self.act(self.linear4(out))+0.0001 # activation
        return torch.cat((out1,out2),dim=-1)

    
class EGLWeightedMSE(torch.nn.Module):
    """
    A weighted version of MSE
    """
    def __init__(self, X, Y, min_val=1e-3):
        super(EGLWeightedMSE, self).__init__()
        # Save true labels
        self.X = X.to('cuda')
        self.Y = Y.to('cuda')
        self.min_val = min_val
        self.X_shape = int(X.numel()/X.shape[0])
        self.Y_shape = int(Y.numel()/Y.shape[0])
        # Initialise paramters
        # self.weights = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))
        self.weights = torch.rand_like(Y[0]).to('cuda')
        self.model = dense_nn(self.X_shape, num_targets=self.weights.numel(), num_layers=4, intermediate_size=500)
        
    def forward(self, data_idx, Yhats):
        if type(data_idx) != int:
            data_idx = list(data_idx)
            Yhats = Yhats[:,int(Yhats.shape[1]/2):]
        else:
            Yhats = Yhats.reshape((1,-1))
        X = self.X[data_idx]
        Y = self.Y[data_idx]
        Xi = X.reshape((-1, self.X_shape)).to('cuda')
        self.weights = self.model(Xi.float())
        # Flatten inputs
        # Yhat = Yhats.view((-1, Y))
        Yi = Y.reshape((-1, self.Y_shape)).to('cuda')

        # Compute MSE
        
        squared_error = (Yhats - Yi).square()
        weighted_mse = (squared_error * self.weights.clamp(min=self.min_val)).mean(dim=-1)

        return weighted_mse


class LCGLN(nn.Module):
    
    def __init__(
        self,
        x_dim,
        y_dim,
        u_dim,
        z_dim,
        output_dim=1,
        act_fn='SOFTPLUS',
        **kwargs
    ):
        super(LCGLN, self).__init__()
        
        if act_fn.upper()=='ELU':
            self.act_fn = nn.ELU()
        elif act_fn.upper()=='SOFTPLUS':
            self.act_fn = nn.Softplus()
        else:
            raise LookupError()
        
        # Input
        #   Upstream
        self.x_to_u = nn.Linear(x_dim, u_dim)
        #   Downstream
        self.x_to_ydim = nn.Linear(x_dim, y_dim)
        self.y_to_zdim = nn.Linear(y_dim, z_dim)
        self.x_to_zdim = nn.Linear(x_dim, z_dim)
        
        # Hidden (for later use)
        #   Upstream
        self.u_to_u = nn.Linear(u_dim, u_dim)
        #   Downstream
        #       1st Term
        self.u_to_zdim = nn.Linear(u_dim, z_dim)
        self.zdim_to_z = nn.Linear(z_dim, z_dim, bias=False)
        #       2nd Term
        self.u_to_ydim = nn.Linear(u_dim, y_dim)
        self.ydim_to_z = nn.Linear(y_dim, z_dim, bias=False)
        #       3rd Term
        self.u_to_z = nn.Linear(u_dim, z_dim)
        
        # Output
        #   Downstream
        #       1st Term
        self.out_u_to_zdim = nn.Linear(u_dim, z_dim)
        self.out_zdim_to_out = nn.Linear(z_dim, output_dim, bias=False)
        #       2nd Term
        self.out_u_to_ydim = nn.Linear(u_dim, y_dim)
        self.out_ydim_to_out = nn.Linear(y_dim, output_dim, bias=False)
        #       3rd Term
        self.out_u_to_out = nn.Linear(u_dim, output_dim)

        
    def forward(self, x, y):
        # Input
        x = -x.float()
        y = -y.float()
        if x.shape == (5, 10):
            x = x.flatten()
        if y.shape == (5, 10):
            y = y.flatten()
        #   Upstream
        u1 = self.x_to_u(x)
        u1 = self.act_fn(u1)
        #   Downstream
        xz1 = self.x_to_zdim(x)
        yz1 = self.y_to_zdim(y)
        z1 = self.act_fn(xz1 + yz1)
        
        # Hidden
        # no hid
        
        # Output
        #   Downstream
        #       1st Term
        uzdim = self.out_u_to_zdim(u1)
        uzdim = torch.clamp_min(uzdim,0) * z1
        uzz2 = self.out_zdim_to_out(uzdim)
        #       2nd Term
        uydim = self.out_u_to_ydim(u1)
        uydim *= y
        uyz2 = self.out_ydim_to_out(uydim)
        #       3rd Term
        uz2 = self.out_u_to_out(u1)
        
        out = self.act_fn(uzz2 + uyz2 + uz2)
        
        return out

   
if __name__ == "__main__":

    pass
