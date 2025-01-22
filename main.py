import argparse
import ast
import csv
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from functools import partial
from tqdm import tqdm
from Dataloader import *
from Networks import dense_nn, dense2_nn
from losses import get_loss_fn
from utils import print_metrics, init_if_not_saved, move_to_gpu
from GBI import GBI

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
    parser.add_argument('--case', type=int, default=1)
    #   Learning Losses
    parser.add_argument('--loss', type=str, choices=['lcgln', 'eglwmse'], default='lcgln')
    parser.add_argument('--serial', type=ast.literal_eval, default=True)
    parser.add_argument('--sampling', type=str, choices=['mbs'], default='mbs')
    parser.add_argument('--numsamples', type=int, default=32)
    parser.add_argument('--losslr', type=float, default=0.001)
    parser.add_argument('--samplinglr', type=float, default=1.0)
    #   LCGLN-specific: Hyperparameters
    parser.add_argument('--lcglnhid', type=int, default=2)
    parser.add_argument('--actfn', type=str, default='SOFTPLUS')
    parser.add_argument('--minmax', type=str, default='MAX')

    #   Domain-specific: Portfolio Optimization
    parser.add_argument('--data_type', type=str, default='Value')
    parser.add_argument('--train_year', type=int, default=20)
    parser.add_argument('--goal', type=int, default=2)
    
    args = parser.parse_args()

    save_folder = os.path.join('case' + str(args.case) + 'goal' + str(args.goal), str(args.loss), str(args.numsamples), str(args.lr, ) + str (args.losslr, ) + str(args.samplinglr))
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
    

    
    print(f"Loading {args.loss} Loss Function...")
    loss_fn = get_loss_fn(
        args.loss,
        problem,
        sampling=args.sampling,
        num_samples=args.numsamples,
        lr=args.losslr,
        serial=args.serial,
        lcgln_hidden_num=args.lcglnhid,
        lcgln_actfn=args.actfn,
        minmax=args.minmax,
        samplinglr=args.samplinglr
    )
        
    # Load an ML model to predict the parameters of the problem
    print(f"Building dense Model...")
    ipdim, opdim = problem.get_modelio_shape()
    model = dense2_nn(
        num_features=ipdim,
        num_targets=opdim,
        intermediate_size=args.intermediatesize
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)     
    
    # Move everything to GPU, if available
    if torch.cuda.is_available():
        move_to_gpu(problem)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
    # Get data [day,stock,feature]
    X_train, Y_train, Y_train_aux = problem.get_train_data()    # [200,50,28], [200,50], [200,50,50]
    X_val, Y_val, Y_val_aux = problem.get_val_data()
    X_test, Y_test, Y_test_aux = problem.get_test_data()

    print('Ours Training Method...')
    if args.minmax.upper()=="MIN":
        best = (float("inf"), None)
    elif args.minmax.upper()=="MAX":
        best = (float("-inf"), None)
    
    else:
        raise LookupError()


    for epoch in tqdm(range(1000)):
        if epoch % args.valfreq == 0:
            # Check if well trained by objective value
            datasets = [(X_train, Y_train, Y_train_aux, 'train'), (X_val, Y_val, Y_val_aux, 'val')]
            # metrics = metrics2wandb(datasets, model, problem, f"Iter {epoch}")
            metrics = print_metrics(datasets, model, problem, args.loss, loss_fn, f"Iter {epoch},", args.wandb)           
            # Save model if it's the best one
            if args.minmax.upper()=="MIN":
                if best[1] is None or metrics['val']['objective'] < best[0]:
                    best = (metrics['val']['objective'], deepcopy(model))
                    steps_since_best = 0
            else:
                if best[1] is None or metrics['val']['objective'] > best[0]:
                    # print(epoch)
                    best = (metrics['val']['objective'], deepcopy(model))
                    steps_since_best = 0

            # Stop if model hasn't improved for patience steps
            if (args.earlystopping) and (steps_since_best > args.patience):
                break
        
        #################### TEST ####################
        # Learn
        def news_loss_fn(pred, y, loss_fn):
            news_losses =[]
            for i in range(y.shape[0]):
                news_losses.append(loss_fn(pred[i], y[i], partition = 'train', index = i))
            return torch.stack(news_losses).mean() 
        
        train_dataloader = GBI_data_loader(X_train, Y_train, batch_size=args.batchsize)
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            X, y = batch
            pred = model(X).squeeze()
            loss = news_loss_fn(pred, y, loss_fn)
            # print(round(loss.item(), 4))
            loss.backward()
            optimizer.step()

        


        # for i in random.sample(range(len(X_train)), min(args.batchsize, len(X_train))):
        #     pred = model(X_train[i].float()).squeeze()
        #     losses.append(loss_fn(pred, Y_train[i], aux_data=Y_train_aux[i], partition='train', index=i))
        # loss = torch.stack(losses).mean()   
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        steps_since_best += 1

        ###############################################       
    if args.earlystopping:
        print("Early Stopping... Saving the Model...")
        model = best[1]      

    # Document how well this trained model does
    print("\nBenchmarking Model...")
    # Print final metrics
    datasets = [(X_train, Y_train, Y_train_aux, 'train'), (X_val, Y_val, Y_val_aux, 'val'), (X_test, Y_test, Y_test_aux, 'test')]
    # metric_ours = metrics2wandb(datasets, model, problem, "Final")
    # metric_ours = print_metrics(datasets, model, problem, args.loss, loss_fn, "Final", args.wandb)   

    #   Document the optimal value
    if args.goal == 1:
        if Y_test.shape[0] != 1:
            for i in range(Y_test.shape[0]):
                Z_test = problem.get_decision(torch.unsqueeze(Y_test[i],dim=0).float(), aux_data=Y_test_aux, isTrain=False)
                Y_pred = torch.unsqueeze(model(X_test[i]),dim=0).detach()
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
            Y_pred = torch.unsqueeze(model(X_test),dim=1).detach()
            Z_pred = problem.get_decision(Y_pred.float(), aux_data=Y_test_aux, isTrain=False)
            objective_opt = problem.get_objective(Y_test, Z_test, Z_test[3:], aux_data=Y_test_aux)[0]
            objectives, infe, g1 = problem.get_objective(Y_test, Z_pred, Z_test[3:], aux_data=Y_test_aux)
        obj_ours = objectives.float().mean().item()
        obj_opt = objectives_opt.mean().item()
        infe_mean = infes / Y_test.shape[0]
        g1_mean = g1s / Y_test.shape[0]
        print(f"Optimal Decision Quality: {obj_opt}")
        
        #   Document OURS value
        # obj_ours = metric_ours['test']['objective']
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
                Y_pred = torch.unsqueeze(model(X_test[i]),dim=0).detach()
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
            Y_pred = torch.unsqueeze(model(X_test),dim=1).detach()
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
        # obj_ours = metric_ours['test']['objective']
        print(f"Our Decision Quality: {obj_ours}")
        print(f"Our feasibility: {infe_mean}")
        print(f"Our goal 1 achieve: {g1_mean}")
        print(f"Our goal 2 achieve: {g2_mean}")

        with open(results_file, 'a') as f:
            f.write('{},{},{},{},{},{},{}\n'.format('DQ_O:', obj_opt, 'DQ:', obj_ours, infe_mean, g1_mean, g2_mean))

        
        pass