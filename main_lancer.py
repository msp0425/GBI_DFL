import argparse
import ast
import numpy as np
import os
import random
import torch

from functools import partial
from tqdm import tqdm
from Dataloader import *
from Networks import dense_nn, dense2_nn
from losses import MSE
from utils import init_if_not_saved, move_to_gpu
from GBI import GBI
from time import time

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, choices=['GBI'], default='GBI')
    parser.add_argument('--loadnew', type=ast.literal_eval, default=True)
    parser.add_argument('--instances', type=int, default=60)
    parser.add_argument('--testinstances', type=int, default=15)
    parser.add_argument('--valfrac', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--intermediatesize', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--earlystopping', type=ast.literal_eval, default=True)
    parser.add_argument('--valfreq', type=int, default=5)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--wandb', type=bool, default=False)
    # parser.add_argument('--twostage', type=bool, default=True)
    parser.add_argument('--case', type=int, default=1)
    #   Learning Losses
    parser.add_argument('--loss', type=str, choices=['lancer'], default='lancer')
    parser.add_argument('--serial', type=ast.literal_eval, default=True)
    parser.add_argument('--numiters', type=int, default=5)
    parser.add_argument('--losslr', type=float, default=0.005)
    parser.add_argument('--samplinglr', type=float, default=0.1)

    #   Domain-specific: Portfolio Optimization
    parser.add_argument('--data_type', type=str, default='Value')
    parser.add_argument('--train_year', type=int, default=20)
    parser.add_argument('--goal', type=int, default=2)
    
    args = parser.parse_args()

    save_folder = os.path.join('case' + str(args.case) + 'goal' + str(args.goal), 'lancer', str(args.numiters), str(args.lr) + str(args.losslr))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    
    results_file = os.path.join(save_folder, f"results.csv")

    # Load problem
    print(f"Hyperparameters: {args}\n")
    print(f"Loading {args.problem} Problem...")
    init_problem = partial(init_if_not_saved, load_new=args.loadnew)
        
    if args.problem == 'GBI':
        problem_kwargs =    {'num_train_instances': args.instances,
                            'num_test_instances': args.testinstances,
                            'data_type': args.data_type,
                            'val_frac': args.valfrac,
                            'train_year': args.train_year,
                            'rand_seed': args.seed,
                            'case': args.case,
                            'goal': args.goal,}
        problem = init_problem(GBI, problem_kwargs)
    
    print(f"Building dense Model...")
    ipdim, opdim = problem.get_modelio_shape()
    model_theta = dense2_nn(
        num_features=ipdim,
        num_targets=opdim,
        intermediate_size=args.intermediatesize
    )
    optimizer_theta = torch.optim.Adam(model_theta.parameters(), lr=args.lr)   

    X_train, Y_train, Y_train_aux = problem.get_train_data()  
    X_val, Y_val, Y_val_aux = problem.get_val_data()
    X_test, Y_test, Y_test_aux = problem.get_test_data()   
    X_train = torch.cat([X_train, X_val], dim=0)
    Y_train = torch.cat([Y_train, Y_val], dim=0)
    Y_train_aux = Y_train_aux + Y_val_aux

    loss_fn = dense_nn(
        num_features=2*int(Y_train.numel()/Y_train.shape[0]),
        num_targets=1,
        num_layers=2,
        intermediate_size=10,
        output_activation=torch.nn.Tanh(),
    )
    optimizer_loss = torch.optim.Adam(loss_fn.parameters(), lr=args.losslr)

    # Move everything to GPU, if available
    if torch.cuda.is_available():
        move_to_gpu(problem)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_theta = model_theta.to(device)
        loss_fn = loss_fn.to(device)
        X_train = X_train.to(device)
        X_test = X_test.to(device)
        Y_train = Y_train.to(device)

    time_start = time()
    sample_set = []
    for t in tqdm(range(args.numiters)):
        samples = model_theta(X_train).squeeze()  
        opt = partial(problem.get_decision, isTrain=True)
        obj = partial(problem.get_objective)
        
        for i in range(Y_train.shape[0]):
            Z = opt(Y_train[i])
            objective = obj(Y_train[i], Z, Z[3:])[0].unsqueeze(dim=0)
            if i == 0:
                opt_objective = objective
            else:
                opt_objective = torch.cat([opt_objective, objective], dim=0)
        
        for i, yhat in enumerate(samples):
            if not i:
                tmp_Zs = opt(yhat)
                tmp_Z_opt = opt(Y_train[i])
                objectives = obj(Y_train[i], tmp_Zs, tmp_Z_opt[3:])[0].unsqueeze(dim=0)
            else:
                tmp_Zs = opt(yhat)
                tmp_Z_opt = opt(Y_train[i])
                objectives = torch.cat((objectives, obj(Y_train[i], tmp_Zs, tmp_Z_opt[3:])[0].unsqueeze(dim=0)), dim=0)
        objectives = opt_objective - objectives
        
        sample_set.append((Y_train, opt_objective, samples.detach(), objectives))
        # Update Surrogate Loss\

        for smp in sample_set:
            y, obj_y, yhat, obj_yhat = smp
            inputs = torch.cat((y, yhat), dim=1).to(device)
            target = obj_yhat.to(device)
            loss_output = loss_fn(inputs.float()).squeeze()
            loss = MSE(loss_output, target.squeeze())
            optimizer_loss.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_loss.step()    
        
        # Update Predictive Model
        for epoch in range(5):
            losses = []
            losses.append( loss_fn( torch.cat((Y_train, model_theta(X_train)), dim=1).float() ) + MSE(Y_train,model_theta(X_train)).float())

            loss = torch.stack(losses).mean()   
            optimizer_theta.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_theta.step()

    time_elapsed = time() - time_start      


    
# Document how well this trained model does
    print("\nBenchmarking Model...")
    # Print final metrics
    datasets = [(X_train, Y_train, Y_train_aux, 'train'), (X_val, Y_val, Y_val_aux, 'val'), (X_test, Y_test, Y_test_aux, 'test')] 

    #   Document the optimal value
    if args.goal == 1:
        if Y_test.shape[0] != 1:
            for i in range(Y_test.shape[0]):
                Z_test = problem.get_decision(torch.unsqueeze(Y_test[i],dim=0).float(), aux_data=Y_test_aux, isTrain=False)
                Y_pred = torch.unsqueeze(model_theta(X_test[i]),dim=0).detach()
                Z_pred = problem.get_decision(Y_pred.float(), aux_data=Y_test_aux, isTrain=False)
                objective_opt = problem.get_objective(torch.unsqueeze(Y_test[i],dim=0).float(), Z_test, Z_test[3:], aux_data=Y_test_aux)[0]
                objective, infe, g1 = problem.get_objective(torch.unsqueeze(Y_test[i],dim=0).float(), Z_pred, Z_test[3:], aux_data=Y_test_aux)
                if i == 0:
                    objectives = objective
                    objectives_opt = objective_opt
                    infes = infe
                    g1s = g1
                else:
                    objectives = torch.cat([objectives, objective], dim=0)
                    objectives_opt = torch.cat([objectives_opt, objective_opt], dim=0)
                    infes += infe
                    g1s += g1
        else:
            Z_test = problem.get_decision(Y_test, aux_data=Y_test_aux, isTrain=False)
            Y_pred = torch.unsqueeze(model_theta(X_test),dim=1).detach()
            Z_pred = problem.get_decision(Y_pred.float(), aux_data=Y_test_aux, isTrain=False)
            objective_opt = problem.get_objective(Y_test, Z_test, Z_test[3:], aux_data=Y_test_aux)[0]
            objectives, infe, g1 = problem.get_objective(Y_test, Z_pred, Z_test[3:], aux_data=Y_test_aux)
        obj_ours = objectives.float().mean().item()
        obj_opt = objectives_opt.mean().item()
        infe_mean = infes / Y_test.shape[0]
        g1_mean = g1s / Y_test.shape[0]
        print(f"Optimal Decision Quality: {obj_opt}")
        
        #   Document OURS value
        print(f"Our Decision Quality: {obj_ours}")
        print(f"Our feasibility: {infe_mean}")
        print(f"Our goal 1 achieve: {g1_mean}")
        

        with open(results_file, 'a') as f:
            f.write('{},{},{},{},{},{}\n'.format('DQ_O:', obj_opt, 'DQ:', obj_ours, infe_mean, g1_mean))

        
        pass
    elif args.goal == 2:
        if Y_test.shape[0] != 1:
            for i in range(Y_test.shape[0]):
                Z_test = problem.get_decision(torch.unsqueeze(Y_test[i],dim=0).float(), aux_data=Y_test_aux, isTrain=False)
                Y_pred = torch.unsqueeze(model_theta(X_test[i]),dim=0).detach()
                Z_pred = problem.get_decision(Y_pred.float(), aux_data=Y_test_aux, isTrain=False)
                objective_opt = problem.get_objective(torch.unsqueeze(Y_test[i],dim=0).float(), Z_test, Z_test[3:], aux_data=Y_test_aux)[0]
                objective, infe, g1, g2 = problem.get_objective(torch.unsqueeze(Y_test[i],dim=0).float(), Z_pred, Z_test[3:], aux_data=Y_test_aux)
                if i == 0:
                    objectives = objective
                    objectives_opt = objective_opt
                    infes = infe
                    g1s = g1
                    g2s = g2
                else:
                    objectives = torch.cat([objectives, objective], dim=0)
                    objectives_opt = torch.cat([objectives_opt, objective_opt], dim=0)
                    infes += infe
                    g1s += g1
                    g2s += g2
        else:
            Z_test = problem.get_decision(Y_test, aux_data=Y_test_aux, isTrain=False)
            Y_pred = torch.unsqueeze(model_theta(X_test),dim=1).detach()
            Z_pred = problem.get_decision(Y_pred.float(), aux_data=Y_test_aux, isTrain=False)
            objective_opt = problem.get_objective(Y_test, Z_test, Z_test[3:], aux_data=Y_test_aux)[0]
            objectives, infe, g1, g2 = problem.get_objective(Y_test, Z_pred, Z_test[3:], aux_data=Y_test_aux)
        obj_ours = objectives.float().mean().item()
        obj_opt = objectives_opt.mean().item()
        infe_mean = infes / Y_test.shape[0]
        g1_mean = g1s / Y_test.shape[0]
        g2_mean = g2s / Y_test.shape[0]
        print(f"Optimal Decision Quality: {obj_opt}")
        
        #   Document OURS value
        print(f"Our Decision Quality: {obj_ours}")
        print(f"Our feasibility: {infe_mean}")
        print(f"Our goal 1 achieve: {g1_mean}")
        print(f"Our goal 2 achieve: {g2_mean}")

        with open(results_file, 'a') as f:
            f.write('{},{},{},{},{},{},{}\n'.format('DQ_O:', obj_opt, 'DQ:', obj_ours, infe_mean, g1_mean, g2_mean))

        
        pass