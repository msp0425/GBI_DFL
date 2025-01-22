from PThenO import PThenO
import pandas as pd
import os
import torch
import cvxpy as cp
import numpy as np

class GBI(PThenO):

    def __init__(
        self,
        num_train_instances=20,  # number of *days* to use from the dataset to train
        num_test_instances=5,  # number of *days* to use from the dataset to test
        val_frac=0.2,  # fraction of training data reserved for test
        rand_seed=0,  # for reproducibility
        data_type='Value',
        train_year=20,
        data_dir='data',
        goal=1,
        case=1
    ):
        super(GBI, self).__init__()
        # Do some random seed fu
        self.rand_seed = rand_seed
        self.train_year = train_year
        self.case = case
        self.goal = goal
        self.params = self.init_GBI_params()
        self._set_seed(self.rand_seed)
        self.data_type = data_type
        # Load train and test labels
        self.Xs_train, self.Ys_train= self._load_instances(data_dir, num_train_instances + num_test_instances, num_test_instances)
        self.Xs_test, self.Ys_test= self._load_instances(data_dir, num_test_instances, 0)
        self.num_asset = int(self.Ys_train.shape[-1]/2)
        self.horizon = self.params['Goal1'].shape[0]
        # Split training data into train/val
        assert 0 < val_frac < 1
        self.val_frac = val_frac
        self.val_idxs = range(0, int(self.val_frac * num_train_instances))
        self.train_idxs = range(int(self.val_frac * num_train_instances), num_train_instances)
        assert all(x is not None for x in [self.train_idxs, self.val_idxs])

        # Undo random seed setting
        self._set_seed()

    def _load_instances(
        self,
        data_dir,
        start,
        end
    ):

        if self.data_type == 'Value':
            data_path = os.path.join(data_dir, "10_Industry_Portfolios_{}.csv".format(1))
        elif self.data_type == 'Equal':
            data_path = os.path.join(data_dir, "10_Industry_Portfolios_{}.csv".format(2))
        data = pd.read_csv(data_path)

        for _ in range(end, start):
            i = 2019 - _
            for j in range(5):
                input_data_mean = torch.tensor(((1+data[(data['Date']>(i+j)*100) & (data['Date']<(i+j+1)*100)].drop('Date',axis=1)/100)**12-1).mean())
                input_data_std = torch.tensor(((1+data[(data['Date']>(i+j)*100) & (data['Date']<(i+j+1)*100)].drop('Date',axis=1)/100)**12-1).std())*0.1
                input_data_cat = torch.cat((input_data_mean, input_data_std),dim=-1)
                if j == 0:
                    input_data = input_data_cat
                else:
                    input_data = torch.cat((input_data, input_data_cat),dim=-1)
            output_data_mean = torch.tensor(((1+data[(data['Date']>(i+5)*100) & (data['Date']<(i+6)*100)].drop('Date',axis=1)/100)**12-1).mean())
            output_data_std = torch.tensor(((1+data[(data['Date']>(i+5)*100) & (data['Date']<(i+6)*100)].drop('Date',axis=1)/100)**12-1).std())*0.1
            output_data = torch.cat((output_data_mean, output_data_std),dim=-1)

            if _ == end:
                Xs = input_data.unsqueeze(dim=0)
                Ys = output_data.unsqueeze(dim=0)
            else:
                Xs = torch.cat((input_data.unsqueeze(dim=0),Xs),dim=0)
                Ys = torch.cat((output_data.unsqueeze(dim=0),Ys),dim=0)


        return Xs, Ys

    def _create_cvxpy_problem(
        self,
        Y
    ):
        if self.goal == 1:
            if len(Y.shape) == 1:
                Y = torch.unsqueeze(Y.float(),dim=0)
            Y = Y.detach().cpu()
            Y[:,self.num_asset:] = Y[:,self.num_asset:] * self.params['sigma']
            ###########################################################################
            R_i = torch.cat([torch.ones(1,10), torch.cumprod((1+ torch.ones([10,1]) @ Y[:,:self.num_asset]),0)], dim = 0)
            r_i_l = torch.clamp(1+Y[:,:self.num_asset]-torch.unsqueeze(self.params['Uncertain'],dim=1) @ Y[:,self.num_asset:], min=0)
            r_i_u = 1+Y[:,:self.num_asset]+torch.unsqueeze(self.params['Uncertain'],dim=1) @ Y[:,self.num_asset:]
            r_f = torch.cumprod((self.params['Rf']+1) * torch.ones([11,10]),axis=0)[:-1,:]
            ###########################################################################
            eta_var = cp.Variable((self.horizon,self.num_asset))
            zeta_var = cp.Variable((self.horizon,self.num_asset))
            xi_var = cp.Variable((self.horizon+1,self.num_asset+1))
            c_var = cp.Variable((self.horizon))
            
            constraints = [eta_var >= 0, 
                        zeta_var >= 0, 
                        xi_var >= 0,
                        c_var >=0,
                        c_var <= self.params['Goal1'],
                        xi_var[1:,:-1] == xi_var[:-1,:-1] - eta_var + zeta_var, 
                        xi_var[0,-1] == 100,
                        xi_var[0,:-1] == 0,
                        cp.max(xi_var[1:,:], axis = 1) <= cp.sum(xi_var[1:, :], axis = 1)*0.2,
                        xi_var[1:,-1] <= xi_var[:-1,-1]
                                            + cp.sum((1-self.params['mu']) * cp.multiply((r_i_l * R_i)[:-1,:], eta_var / r_f)
                                            - (1+self.params['nu']) * cp.multiply((r_i_u * R_i)[:-1,:], zeta_var / r_f),axis=1)
                                            - (c_var/torch.cumprod(((self.params['Rf']+1) * torch.ones([10])),axis=0))]
            objective = cp.Maximize( cp.sum(c_var/torch.cumprod(((self.params['Rf']+1) * torch.ones([10])),axis=0)))
            problem = cp.Problem(objective, constraints)

            result_1 = problem.solve(solver=cp.ECOS)
            c_0 = c_var.value


            eta_var2 = cp.Variable((self.horizon,self.num_asset))
            zeta_var2 = cp.Variable((self.horizon,self.num_asset))
            xi_var2 = cp.Variable((self.horizon+1,self.num_asset+1))
            c_var2 = cp.Variable((self.horizon))
            c_var1 = torch.tensor(c_0)

            constraints2 = [eta_var2 >= 0, 
                        zeta_var2 >= 0, 
                        xi_var2 >= 0,
                        c_var2 <= self.params['Goal2'] + c_var1,
                        c_var2 >= c_var1,
                        xi_var2[1:,:-1] == xi_var2[:-1,:-1] - eta_var2 + zeta_var2, 
                        xi_var2[0,-1] == 100,
                        xi_var2[0,:-1] == 0,
                        cp.max(xi_var2[1:,:], axis = 1) <= cp.sum(xi_var2[1:, :], axis = 1)*0.2,
                        xi_var2[1:,-1] <= xi_var2[:-1,-1]
                                            + cp.sum((1-self.params['mu']) * cp.multiply((r_i_l * R_i)[:-1,:], eta_var2 / r_f)
                                            - (1+self.params['nu']) * cp.multiply((r_i_u * R_i)[:-1,:], zeta_var2 / r_f),axis=1)
                                            - (c_var2/torch.cumprod(((self.params['Rf']+1) * torch.ones([10])),axis=0))]
            objective2 = cp.Maximize( cp.sum(c_var2/torch.cumprod(((self.params['Rf']+1) * torch.ones([10])),axis=0)))
            problem2 = cp.Problem(objective2, constraints2)  

            result_2 = problem2.solve(solver=cp.ECOS)
            c_2 = torch.tensor(c_var2.value)



            return result_2 , c_var2.value, eta_var2.value, zeta_var2.value, xi_var2.value, c_var1,  c_2
        elif self.goal == 2:
            if len(Y.shape) == 1:
                Y = torch.unsqueeze(Y.float(),dim=0)
            Y = Y.detach().cpu()
            Y[:,self.num_asset:] = Y[:,self.num_asset:] * self.params['sigma']
            ###########################################################################
            R_i = torch.cat([torch.ones(1,10), torch.cumprod((1+ torch.ones([10,1]) @ Y[:,:self.num_asset]),0)], dim = 0)
            r_i_l = torch.clamp(1+Y[:,:self.num_asset]-torch.unsqueeze(self.params['Uncertain'],dim=1) @ Y[:,self.num_asset:], min=0)
            r_i_u = 1+Y[:,:self.num_asset]+torch.unsqueeze(self.params['Uncertain'],dim=1) @ Y[:,self.num_asset:]
            r_f = torch.cumprod((self.params['Rf']+1) * torch.ones([11,10]),axis=0)[:-1,:]
            ###########################################################################
            eta_var = cp.Variable((self.horizon,self.num_asset))
            zeta_var = cp.Variable((self.horizon,self.num_asset))
            xi_var = cp.Variable((self.horizon+1,self.num_asset+1))
            c_var = cp.Variable((self.horizon))
            
            constraints = [eta_var >= 0, 
                        zeta_var >= 0, 
                        xi_var >= 0,
                        c_var >=0,
                        c_var <= self.params['Goal1'],
                        xi_var[1:,:-1] == xi_var[:-1,:-1] - eta_var + zeta_var, 
                        xi_var[0,-1] == 100,
                        xi_var[0,:-1] == 0,
                        cp.max(xi_var[1:,:], axis = 1) <= cp.sum(xi_var[1:, :], axis = 1)*0.2,
                        xi_var[1:,-1] <= xi_var[:-1,-1]
                                            + cp.sum((1-self.params['mu']) * cp.multiply((r_i_l * R_i)[:-1,:], eta_var / r_f)
                                            - (1+self.params['nu']) * cp.multiply((r_i_u * R_i)[:-1,:], zeta_var / r_f),axis=1)
                                            - (c_var/torch.cumprod(((self.params['Rf']+1) * torch.ones([10])),axis=0))]
            objective = cp.Maximize( cp.sum(c_var/torch.cumprod(((self.params['Rf']+1) * torch.ones([10])),axis=0)))
            problem = cp.Problem(objective, constraints)

            result_1 = problem.solve(solver=cp.ECOS)
            c_0 = c_var.value


            eta_var2 = cp.Variable((self.horizon,self.num_asset))
            zeta_var2 = cp.Variable((self.horizon,self.num_asset))
            xi_var2 = cp.Variable((self.horizon+1,self.num_asset+1))
            c_var2 = cp.Variable((self.horizon))
            c_var1 = torch.tensor(c_0)

            constraints2 = [eta_var2 >= 0, 
                        zeta_var2 >= 0, 
                        xi_var2 >= 0,
                        c_var2 <= self.params['Goal2'] + c_var1,
                        c_var2 >= c_var1,
                        xi_var2[1:,:-1] == xi_var2[:-1,:-1] - eta_var2 + zeta_var2, 
                        xi_var2[0,-1] == 100,
                        xi_var2[0,:-1] == 0,
                        cp.max(xi_var2[1:,:], axis = 1) <= cp.sum(xi_var2[1:, :], axis = 1)*0.2,
                        xi_var2[1:,-1] <= xi_var2[:-1,-1]
                                            + cp.sum((1-self.params['mu']) * cp.multiply((r_i_l * R_i)[:-1,:], eta_var2 / r_f)
                                            - (1+self.params['nu']) * cp.multiply((r_i_u * R_i)[:-1,:], zeta_var2 / r_f),axis=1)
                                            - (c_var2/torch.cumprod(((self.params['Rf']+1) * torch.ones([10])),axis=0))]
            objective2 = cp.Maximize( cp.sum(c_var2/torch.cumprod(((self.params['Rf']+1) * torch.ones([10])),axis=0)))
            problem2 = cp.Problem(objective2, constraints2)  

            result_2 = problem2.solve(solver=cp.ECOS)
            c_2 = c_var2.value



            eta_var3 = cp.Variable((self.horizon,self.num_asset))
            zeta_var3 = cp.Variable((self.horizon,self.num_asset))
            xi_var3 = cp.Variable((self.horizon+1,self.num_asset+1))
            c_var3 = cp.Variable((self.horizon))
            c_var2 = torch.tensor(c_2)

            constraints3 = [eta_var3 >= 0, 
                        zeta_var3 >= 0, 
                        xi_var3 >= 0,
                        c_var3 <= self.params['Goal3'] + c_var2,
                        c_var3 >= c_var2,
                        xi_var3[1:,:-1] == xi_var3[:-1,:-1] - eta_var3 + zeta_var3, 
                        xi_var3[0,-1] == 100,
                        xi_var3[0,:-1] == 0,
                        cp.max(xi_var3[1:,:], axis = 1) <= cp.sum(xi_var3[1:, :], axis = 1)*0.2,
                        xi_var3[1:,-1] <= xi_var3[:-1,-1]
                                            + cp.sum((1-self.params['mu']) * cp.multiply((r_i_l * R_i)[:-1,:], eta_var3 / r_f)
                                            - (1+self.params['nu']) * cp.multiply((r_i_u * R_i)[:-1,:], zeta_var3 / r_f),axis=1)
                                            - (c_var3/torch.cumprod(((self.params['Rf']+1) * torch.ones([10])),axis=0))]
            objective3 = cp.Maximize( cp.sum(c_var3/torch.cumprod(((self.params['Rf']+1) * torch.ones([10])),axis=0)))
            problem3 = cp.Problem(objective3, constraints3)  

            result_3 = problem3.solve(solver=cp.ECOS)
            c_3 = torch.tensor(c_var3.value)


            return result_3 , c_var3.value, eta_var3.value, zeta_var3.value, xi_var3.value, c_var1, c_var2
    
    def init_GBI_params(self):
        params = {}

        # Ordering costs
        params['mu'] = 0.05
        params['nu'] = 0.05

        # Goal amounts
        if self.goal == 1:
            params['Goal1'] = torch.tensor([0, 10, 0, 0, 50, 0, 0, 0, 0, 0])
            params['Goal2'] = torch.tensor([0, 0, 20, 0, 0, 0, 0, 0, 200, 0])
        elif self.goal == 2:
            params['Goal1'] = torch.tensor([0, 10, 0, 0, 0, 30, 0, 0, 0, 0])
            params['Goal2'] = torch.tensor([0, 0, 0, 0, 30, 0, 10, 0, 0, 0])
            params['Goal3'] = torch.tensor([0, 0, 10, 0, 0, 0, 0, 0, 150, 0])

        # Invest amount
        params['Invest'] = torch.tensor([100, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        if self.case == 1:
            params['Uncertain'] = torch.tensor(np.cumprod(np.full(11, 1))/1).float()
            params['sigma'] = 1
        elif self.case == 2:
            params['Uncertain'] = torch.tensor(np.cumprod(np.full(11, 1.2))/1.2).float()
            params['sigma'] = 1
        elif self.case == 3:
            params['Uncertain'] = torch.tensor(np.cumprod(np.full(11, 1.2))/1.2).float()
            params['sigma'] = 2

        # Risk free
        params['Rf'] = torch.tensor(0.01)

        return params
    
    def get_train_data(self):
        return self.Xs_train[self.train_idxs], self.Ys_train[self.train_idxs],  [None for _ in range(len(self.train_idxs))]

    def get_val_data(self):
        return self.Xs_train[self.val_idxs], self.Ys_train[self.val_idxs],  [None for _ in range(len(self.val_idxs))]

    def get_test_data(self):
        return self.Xs_test, self.Ys_test,  [None for _ in range(len(self.Ys_test))]

    def get_modelio_shape(self):
        return self.Xs_train.shape[-1], self.Ys_train.shape[-1]

    def get_twostageloss(self):
        return 'mse'

    def get_decision(self, Y, **kwargs):
        return self._create_cvxpy_problem(Y)[2:]

    def get_objective(self, Y, Z, c, **kwargs):
        if self.goal == 1:
            c_0_op = c
            infe = 1
            g1 = 1
            Y = Y.cpu()
            if len(Y.shape) == 1:
                Y = torch.unsqueeze(Y.float(),dim=0)
            Y[:,self.num_asset:] = Y[:,self.num_asset:] * self.params['sigma']
            ###########################################################################
            R_i = torch.cat([torch.ones(1,10), torch.cumprod((1+ torch.ones([10,1]) @ Y[:,:self.num_asset].float()),0)], dim = 0)
            r_i_l = torch.clamp(1+Y[:,:self.num_asset]-torch.unsqueeze(self.params['Uncertain'],dim=1) @ Y[:,self.num_asset:], min=0)
            r_i_u = 1+Y[:,:self.num_asset]+torch.unsqueeze(self.params['Uncertain'],dim=1) @ Y[:,self.num_asset:]
            r_f = torch.cumprod((self.params['Rf']+1) * torch.ones([11,10]),axis=0)[:-1,:]
            ###########################################################################
            eta_var3, zeta_var3, xi_var3_ , c_0_, c_1_= Z
            eta_var3 = torch.tensor(eta_var3)
            zeta_var3 = torch.tensor(zeta_var3)
            xi_var3_ = torch.tensor(xi_var3_)
            c_var3 = cp.Variable((self.horizon))
            xi_var3 = cp.Variable((self.horizon+1,self.num_asset+1))

            constraints3 = [c_var3 <= self.params['Goal1'],
                            c_var3 >= 0,
                            xi_var3[:,:-1] == xi_var3_[:,:-1],
                            xi_var3[0, -1] == 100,
                            xi_var3 >= 0,
                            xi_var3[1:,-1] <= xi_var3[:-1,-1]
                                            + torch.sum((1-self.params['mu']) * (r_i_l * R_i)[:-1,:] * eta_var3 / r_f
                                            - (1+self.params['nu']) * (r_i_u * R_i)[:-1,:] * zeta_var3 / r_f,axis=1)
                                            - (c_var3/torch.cumprod(((self.params['Rf']+1) * torch.ones([10])),axis=0))]
            objective3 = cp.Maximize( cp.sum(c_var3/torch.cumprod((self.params['Rf']+1) * torch.ones([10]),axis=0)))
            problem3 = cp.Problem(objective3, constraints3)        
            result_3 = problem3.solve(solver=cp.MOSEK)
            if problem3.status == 'infeasible':
                infe = 0
                g1 = 0
                return torch.tensor([0]), infe, g1
            elif problem3.status.upper() == 'UNKNOWN':
                infe = 0
                g1 = 0
                return torch.tensor([0]), infe, g1

            c_1 = c_var3.value
            if c_1 is None:
                infe = 0
                g1 = 0
                return torch.tensor([0]), infe, g1
            elif (np.round(np.array(c_0_op),4) - np.round(c_1,4) !=0).sum() != 0:
                g1 = 0
                return torch.unsqueeze(torch.tensor(result_3),dim=0), infe, g1
            
            c_var4 = cp.Variable((self.horizon))
            xi_var4 = cp.Variable((self.horizon+1,self.num_asset+1))
            constraints4 = [c_var4 <= self.params['Goal2'] + c_1,
                            c_var4 >= c_1,
                            xi_var4[:,:-1] == xi_var3_[:,:-1],
                            xi_var4[0, -1] == 100,
                            xi_var4 >= 0,
                            xi_var4[1:,-1] <= xi_var4[:-1,-1]
                                            + torch.sum((1-self.params['mu']) * (r_i_l * R_i)[:-1,:] * eta_var3/r_f
                                            - (1+self.params['nu']) * (r_i_u * R_i)[:-1,:] * zeta_var3 / r_f,axis=1)
                                            - (c_var4/torch.cumprod(((self.params['Rf']+1) * torch.ones([10])),axis=0))]
            objective4 = cp.Maximize( cp.sum(c_var4/torch.cumprod((self.params['Rf']+1) * torch.ones([10]),axis=0)))
            problem4 = cp.Problem(objective4, constraints4)        
            result_4 = problem4.solve(solver=cp.ECOS)

            return torch.unsqueeze(torch.tensor(result_4),dim=0), infe, g1
        
        elif self.goal == 2:
            c_0_op, c_2_op = c
            infe = 1
            g1 = 1
            g2 = 1
            Y = Y.cpu()
            if len(Y.shape) == 1:
                Y = torch.unsqueeze(Y.float(),dim=0)
            Y[:,self.num_asset:] = Y[:,self.num_asset:] * self.params['sigma']
            ###########################################################################
            R_i = torch.cat([torch.ones(1,10), torch.cumprod((1+ torch.ones([10,1]) @ Y[:,:self.num_asset].float()),0)], dim = 0)
            r_i_l = torch.clamp(1+Y[:,:self.num_asset]-torch.unsqueeze(self.params['Uncertain'],dim=1) @ Y[:,self.num_asset:], min=0)
            r_i_u = 1+Y[:,:self.num_asset]+torch.unsqueeze(self.params['Uncertain'],dim=1) @ Y[:,self.num_asset:]
            r_f = torch.cumprod((self.params['Rf']+1) * torch.ones([11,10]),axis=0)[:-1,:]
            ###########################################################################
            eta_var3, zeta_var3, xi_var3_ , c_0_, c_1_= Z
            eta_var3 = torch.tensor(eta_var3)
            zeta_var3 = torch.tensor(zeta_var3)
            xi_var3_ = torch.tensor(xi_var3_)
            c_var3 = cp.Variable((self.horizon))
            xi_var3 = cp.Variable((self.horizon+1,self.num_asset+1))

            constraints3 = [c_var3 <= self.params['Goal1'],
                            c_var3 >= 0,
                            xi_var3[:,:-1] == xi_var3_[:,:-1],
                            xi_var3[0, -1] == 100,
                            xi_var3 >= 0,
                            xi_var3[1:,-1] <= xi_var3[:-1,-1]
                                            + torch.sum((1-self.params['mu']) * (r_i_l * R_i)[:-1,:] * eta_var3 / r_f
                                            - (1+self.params['nu']) * (r_i_u * R_i)[:-1,:] * zeta_var3 / r_f,axis=1)
                                            - (c_var3/torch.cumprod(((self.params['Rf']+1) * torch.ones([10])),axis=0))]
            objective3 = cp.Maximize( cp.sum(c_var3/torch.cumprod((self.params['Rf']+1) * torch.ones([10]),axis=0)))
            problem3 = cp.Problem(objective3, constraints3)        
            result_3 = problem3.solve(solver=cp.MOSEK)
            if problem3.status == 'infeasible':
                infe = 0
                g1 = 0
                g2 = 0
                return torch.tensor([0]), infe, g1, g2
            elif problem3.status.upper() == 'UNKNOWN':
                infe = 0
                g1 = 0
                g2 = 0
                return torch.tensor([0]), infe, g1, g2

            c_1 = c_var3.value
            if c_1 is None:
                infe = 0
                g1 = 0
                g2 = 0
                return torch.tensor([0]), infe, g1, g2
            elif (np.round(np.array(c_0_op),4) - np.round(c_1,4) !=0).sum() != 0:
                g1 = 0
                g2 = 0
                return torch.unsqueeze(torch.tensor(result_3),dim=0), infe, g1, g2
            
            c_var4 = cp.Variable((self.horizon))
            xi_var4 = cp.Variable((self.horizon+1,self.num_asset+1))
            constraints4 = [c_var4 <= self.params['Goal2'] + c_1,
                            c_var4 >= c_1,
                            xi_var4[:,:-1] == xi_var3_[:,:-1],
                            xi_var4[0, -1] == 100,
                            xi_var4 >= 0,
                            xi_var4[1:,-1] <= xi_var4[:-1,-1]
                                            + torch.sum((1-self.params['mu']) * (r_i_l * R_i)[:-1,:] * eta_var3/r_f
                                            - (1+self.params['nu']) * (r_i_u * R_i)[:-1,:] * zeta_var3 / r_f,axis=1)
                                            - (c_var4/torch.cumprod(((self.params['Rf']+1) * torch.ones([10])),axis=0))]
            objective4 = cp.Maximize( cp.sum(c_var4/torch.cumprod((self.params['Rf']+1) * torch.ones([10]),axis=0)))
            problem4 = cp.Problem(objective4, constraints4)        
            result_4 = problem4.solve(solver=cp.ECOS)

            c_2 = c_var4.value

            if (np.round(np.array(c_2_op),4) - np.round(c_2,4) !=0).sum() != 0:
                g2 = 0
                return torch.unsqueeze(torch.tensor(result_3),dim=0), infe, g1, g2

            c_var5 = cp.Variable((self.horizon))
            xi_var5 = cp.Variable((self.horizon+1,self.num_asset+1))
            constraints5 = [c_var5 <= self.params['Goal3'] + c_2,
                            c_var5 >= c_2,
                            xi_var5[:,:-1] == xi_var3_[:,:-1],
                            xi_var5[0, -1] == 100,
                            xi_var5 >= 0,
                            xi_var5[1:,-1] <= xi_var5[:-1,-1]
                                            + torch.sum((1-self.params['mu']) * (r_i_l * R_i)[:-1,:] * eta_var3/r_f
                                            - (1+self.params['nu']) * (r_i_u * R_i)[:-1,:] * zeta_var3 / r_f,axis=1)
                                            - (c_var5/torch.cumprod(((self.params['Rf']+1) * torch.ones([10])),axis=0))]
            objective5 = cp.Maximize( cp.sum(c_var5/torch.cumprod((self.params['Rf']+1) * torch.ones([10]),axis=0)))
            problem5 = cp.Problem(objective5, constraints5)        
            result_5 = problem5.solve(solver=cp.ECOS)            



            return torch.unsqueeze(torch.tensor(result_5),dim=0), infe, g1 , g2

    def get_output_activation(self):
        return 'relu'

