import os
import pickle
import random
import time
import torch
import numpy as np
from copy import deepcopy
from functools import partial
from torch.nn.functional import mse_loss
from Networks import dense_nn, dense2_nn
from Dataloader import *
from Networks import LCGLN, EGLWeightedMSE
from utils import find_saved_problem


NUM_CPUS = os.cpu_count()


def MSE(Yhats, Ys, **kwargs):
    """
    Calculates the mean squared error between predictions
    Yhat and true lables Y.
    """
    return (Yhats - Ys).square().mean()


def MAE(Yhats, Ys, **kwargs):
    """
    Calculates the mean squared error between predictions
    Yhat and true lables Y.
    """
    return (Yhats - Ys).abs().mean()


def CE(Yhats, Ys, **kwargs):
    return torch.nn.BCELoss()(Yhats, Ys)

def MSE_Sum(
    Yhats,
    Ys,
    alpha=0.1,  # weight of MSE-based regularisation
    **kwargs
):

    # Check if prediction is a matrix/tensor
    assert len(Ys.shape) >= 2

    # Calculate loss
    sum_loss = (Yhats - Ys).sum(dim=-1).square().mean()
    loss_regularised = (1 - alpha) * sum_loss + alpha * MSE(Yhats, Ys)
    return loss_regularised


def _learn_loss(
    problem,  # The problem domain
    dataset,  # The data set on which to train SL
    model_type,  # The model we're trying to fit
    num_iters=100,  # Number of iterations over which to train model
    lr=1,  # Learning rate with which to train the model
    verbose=False,  # print training loss?
    train_frac=0.3,  # fraction of samples to use for training
    val_frac=0.3,  # fraction of samples to use for testing
    val_freq=1,  # the number of training steps after which to check loss on val set
    print_freq=5,  # the number of val steps after which to print losses
    patience=10,  # number of iterations to wait for the train loss to improve when learning
    lcgln_actfn='ELU',
    minmax='MIN',
    data_idx=None,
    **kwargs
):
    """
    Function that learns a model to approximate the behaviour of the
    'decision-focused loss' from Wilder et. al. in the neighbourhood of Y
    """
    
    # Get samples from dataset
    Y, opt_objective, Yhats, objectives = dataset
    if minmax.upper()=="MIN":
        objectives = objectives - opt_objective
    else:
        objectives = opt_objective - objectives

    assert train_frac + val_frac < 1
    
    # Split train and test  
    train_idxs = range(0, int(train_frac * Yhats.shape[0]))
    val_idxs = range(int(train_frac * Yhats.shape[0]), int((train_frac + val_frac) * Yhats.shape[0]))
    test_idxs = range(int((train_frac + val_frac) * Yhats.shape[0]), Yhats.shape[0])

    if model_type.upper()=="GICNN" or model_type.upper() == 'EGLWMSE' or model_type.upper() == 'EGLDQ': 
        Yhats_train, objectives_train, data_idx_train = Yhats[train_idxs], objectives[train_idxs], data_idx[train_idxs]
        Yhats_val, objectives_val, data_idx_val = Yhats[val_idxs], objectives[val_idxs], data_idx[val_idxs]
        Yhats_test, objectives_test, data_idx_test = Yhats[test_idxs], objectives[test_idxs], data_idx[test_idxs]
    else:
        Yhats_train, objectives_train = Yhats[train_idxs], objectives[train_idxs]
        Yhats_val, objectives_val = Yhats[val_idxs], objectives[val_idxs]
        Yhats_test, objectives_test = Yhats[test_idxs], objectives[test_idxs]
    
    # Load a model
    if model_type.upper() == 'LCGLN':
        tmp = int(Yhats.shape[-1]/2)
        model = LCGLN(x_dim=tmp, y_dim=tmp, u_dim=tmp, z_dim=tmp, act_fn=lcgln_actfn)
    elif model_type.upper() == 'EGLWMSE':
        model = EGLWeightedMSE(problem.get_train_data()[0], Y, **kwargs)
    else:
        raise LookupError()
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Yhats_train, Yhats_val, Yhats_test = Yhats_train.to(device), Yhats_val.to(device), Yhats_test.to(device)
        objectives_train, objectives_val, objectives_test = objectives_train.to(device), objectives_val.to(device), objectives_test.to(device)
        model = model.to(device)

    
    # Fit a model to the points
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best = (float("inf"), None)
    time_since_best = 0

    if (model_type.upper() == 'LCGLN'):
        # Generate Dataloader
        batch_size = 16
        n_epochs = 1000
        train_dataloader = GBI_data_loader(Yhats_train, objectives_train, batch_size=batch_size)
        val_dataloader = GBI_data_loader(Yhats_val, objectives_val, batch_size=batch_size)
        lcgln = True if model_type.upper() == "LCGLN" else False

        time_since_best = 0
        for iter_idx in range(n_epochs):
            train_loss_tracker = []
            val_loss_tracker = []
            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                X, y = batch
                X = X.to(device)
                pred = model(X[:, :tmp], X[:, tmp:]) if lcgln else model(X)
                # pred = model(X)
                loss = mse_loss(pred, y.float())
                train_loss_tracker.append(loss.item())
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    X_val, y_val = batch
                    X_val = X_val.to(device)
                    y_val = y_val.to(device)
                    pred_val = model(X_val[ : , :tmp], X_val[ : , tmp: ]).flatten() if lcgln else model(X_val).flatten()
                    loss_val = MSE(pred_val, y_val)
                    val_loss_tracker.append(loss_val.item())
                
            if verbose and iter_idx % (val_freq * print_freq) == 0:
                print(f"Iter {iter_idx}, Train Loss MSE: {np.mean(train_loss_tracker)}")
                print(f"Iter {iter_idx}, Val Loss MSE: {np.mean(val_loss_tracker)}")
            
            if best[1] is None or np.mean(val_loss_tracker) < best[0]:
                best = (np.mean(val_loss_tracker), deepcopy(model))
                time_since_best = 0
            else:
                time_since_best += 1
                
            if time_since_best > patience:
                break
        
    else:
        for iter_idx in range(num_iters):
            # Define update step using "closure" function
            def loss_closure():
                optimizer.zero_grad()
                if model_type.upper()=='EGLWMSE' or model_type.upper()=='EGLDQ':
                    pred = model(data_idx_train, Yhats_train).flatten()
                else:
                    pred = model(Yhats_train).flatten()
                # if not (pred >= -1e-3).all().item():
                #     print(f"WARNING: Prediction value < 0: {pred.min()}")
                loss = MSE(pred, objectives_train)
                loss.backward()
                # print(round(loss.item(), 4))
                return loss

            # Perform validation
            if iter_idx % val_freq == 0:
                # Get performance on val dataset
                if model_type.upper()=='EGLWMSE' or model_type.upper()=='EGLDQ':
                    pred_val = model(data_idx_val, Yhats_val).flatten()
                else:
                    pred_val = model(Yhats_val).flatten()
                loss_val = MSE(pred_val, objectives_val)

                # Print statistics
                if verbose and iter_idx % (val_freq * print_freq) == 0:
                    print(f"Iter {iter_idx}, Train Loss MSE: {loss_closure().item()}")
                    print(f"Iter {iter_idx}, Val Loss MSE: {loss_val.item()}")
                # Save model if it's the best one
                if best[1] is None or loss_val.item() < best[0]:
                    # best = (loss_val.item(), deepcopy(model))
                    best = (loss_val.item(), model)
                    time_since_best = 0
                # Stop if model hasn't improved for patience steps
                if time_since_best > patience:
                    break

            # Make an update step
            optimizer.step(loss_closure)
            time_since_best += 1
            
    model = best[1]
    
    return model, 0, 0


def _get_learned_loss(
    problem,
    model_type='weightedmse',
    folder='models',
    num_samples=400,
    sampling='mbs',
    minmax='MIN',
    sampling_lr=0,     
    **kwargs
):
    # Learn Losses
    #   Get Ys
    X_train, Y_train, Y_train_aux = problem.get_train_data()
    X_val, Y_val, Y_val_aux = problem.get_val_data()

    master_filename = os.path.join(folder, f"{problem.__class__.__name__}.csv")
    problem_filename, _ = find_saved_problem(master_filename, problem.__dict__)
    samples_filename_read = f"{problem_filename[:-4]}_{sampling}_{sampling_lr}_{num_samples}-.pkl"


    # Check if there are enough stored samples
    num_samples_needed = num_extra_samples = num_samples
    if os.path.exists(samples_filename_read):
        with open(samples_filename_read, 'rb') as filehandle:
            num_existing_samples, SL_dataset_old = pickle.load(filehandle)
    else:
        num_existing_samples = 0
        SL_dataset_old = {partition: [(Y, None, None, None) for Y in Ys] for Ys, partition in zip([Y_train, Y_val], ['train', 'val'])}

    # Sample more points if needed
    num_samples_needed = num_samples
    num_extra_samples = max(num_samples_needed - num_existing_samples, 0)
    datasets = [entry for entry in zip([Y_train, Y_val], [Y_train_aux, Y_val_aux], ['train', 'val'])]
    if num_extra_samples > 0:
        SL_dataset = {partition: [(Y, None, None, None) for Y in Ys] for Ys, partition in zip([Y_train, Y_val], ['train', 'val'])}
        for Ys, Ys_aux, partition in datasets:
            
            print(f'(smp_lr={sampling_lr}) Training dense model for mbs inside losses.py...')
            if partition=='train':
                Xs = X_train.clone()
            elif partition=='val':
                Xs = X_val.clone()
            else:
                raise TypeError
            ipdim, opdim = problem.get_modelio_shape()
            sampling_model = dense2_nn(
                    num_features=ipdim,
                    num_targets=opdim,
                    intermediate_size=10,
                )
            optimizer = torch.optim.Adam(sampling_model.parameters(), lr=sampling_lr)
            # val_loss = 10e6
    
            Yhats = Ys.clone()
            # for epoch in range(300):
            for epoch in range(num_extra_samples+1):
                loss = MSE(sampling_model(Xs).squeeze(), Ys)
                sampling_freq = max(1, int(300/num_extra_samples))

                if epoch>=2:
                    print(loss)

                    Yhats = torch.cat((Yhats, sampling_model(Xs)), dim=0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            opt = partial(problem.get_decision, isTrain=False, aux_data=Ys_aux)
            obj = partial(problem.get_objective, aux_data=Ys_aux)

            for i in range(Ys.shape[0]):
                Z = opt(Ys[i])
                objective = obj(Ys[i], Z, Z[3:])[0].unsqueeze(dim=0)
                if i == 0:
                    opt_objective = objective
                else:
                    opt_objective = torch.cat([opt_objective, objective], dim=0)
            opt_objective = torch.unsqueeze(opt_objective,dim=1)

            for i, yhat in enumerate(Yhats):
                if not i:
                    tmp_Zs = opt(yhat)
                    tmp_Z_opt = opt(Ys[i%Ys.shape[0]])
                    objectives = obj(Ys[i%Ys.shape[0]], tmp_Zs, tmp_Z_opt[3:])[0].unsqueeze(dim=0).unsqueeze(dim=0)
                else:
                    tmp_Zs = opt(yhat)
                    tmp_Z_opt = opt(Ys[i%Ys.shape[0]])
                    objectives = torch.cat((objectives, obj(Ys[i%Ys.shape[0]], tmp_Zs, tmp_Z_opt[3:])[0].unsqueeze(dim=0).unsqueeze(dim=0)), dim=0)

            sampled_points = []
            for i in range(Ys.shape[0]):
                sampled_points.append((Ys[i].detach(), opt_objective[i].squeeze().detach(), Yhats[torch.tensor(range(Yhats.shape[0]))%Ys.shape[0] == i].detach(), objectives[torch.tensor(range(Yhats.shape[0]))%Ys.shape[0] == i, 0].detach()))
    

            # Use them to augment existing sampled points
            for idx, (Y, opt_objective, Yhats, objectives) in enumerate(sampled_points):
                SL_dataset[partition][idx] = (Y, opt_objective, Yhats, objectives)



        #   Augment with new data
        for Ys, Ys_aux, partition in datasets:
            for idx, Y in enumerate(Ys):
                # Get old samples
                Y_old, opt_objective_old, Yhats_old, objectives_old = SL_dataset_old[partition][idx]
                Y_new, opt_objective_new, Yhats_new, objectives_new = SL_dataset[partition][idx]
                assert torch.isclose(Y_old, Y).all()
                assert torch.isclose(Y_new, Y).all()

                # Combine entries
                opt_objective = opt_objective_new if opt_objective_old is None else max(opt_objective_new, opt_objective_old)
                Yhats = Yhats_new if Yhats_old is None else torch.cat((Yhats_old, Yhats_new), dim=0)
                objectives = objectives_new if objectives_old is None else torch.cat((objectives_old, objectives_new), dim=0)

                # Update
                SL_dataset[partition][idx] = (Y, opt_objective, Yhats, objectives)
        num_existing_samples += num_extra_samples

    else:
        print("Loading from Saved Sample Data...")
        SL_dataset = SL_dataset_old

    print("Learning Loss Functions...")

    #   Learn SL based on the sampled Yhats
    losses = {}
    if model_type.upper()=='LCGLN' or model_type.upper()=='EGLWMSE':
        for Ys, Ys_aux, partition in [datasets[0]]:
            
            Y_Yhats_ = [ torch.cat( (tmp[0].flatten().repeat(tmp[2].shape[0],1), tmp[2].flatten(1)), dim=1 ) for tmp in SL_dataset[partition] ]
            Y_Yhats = torch.vstack(Y_Yhats_)
            opt_objs = torch.vstack( [ tmp[1].repeat(tmp[2].shape[0],1) for tmp in SL_dataset['train'] ] )
            objs = torch.vstack( [ tmp[3] for tmp in SL_dataset['train'] ] )
            data_idx = torch.tensor(np.repeat(range(Ys.shape[0]),SL_dataset['train'][0][3].shape[0]))
            random.seed(0) 
            idxs = random.sample(range(Y_Yhats.shape[0]), Ys.shape[0]*num_samples_needed)
            random.seed()
            
            start_time = time.time()
            losses[partition] = _learn_loss(problem, (Ys, opt_objs[idxs], Y_Yhats[idxs], objs[idxs]), model_type, minmax=minmax, data_idx=data_idx[idxs], **kwargs)
            print(f"({partition}) Time taken to learn loss for {len(Ys)} instances: {round(time.time() - start_time, 2)} sec")
        

    def surrogate_decision_quality(Yhats, Ys, partition, index, **kwargs):
        if model_type.upper() == 'LCGLN':
            return losses[partition][0](Ys, Yhats).flatten()
        elif model_type.upper() == 'EGLWMSE':
            return losses[partition][0](index, Yhats).flatten()
        else:
            raise LookupError()
    return surrogate_decision_quality

def get_loss_fn(
    name,
    problem,
    samplinglr = 0,
    **kwargs
):
    return _get_learned_loss(problem, name, sampling_lr = samplinglr, **kwargs)
