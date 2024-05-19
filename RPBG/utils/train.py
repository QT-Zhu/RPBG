import os
import numpy as np
import torch
import gzip
import pydoc

from RPBG.models.compose import ModelAndLoss


def to_device(data, device='cuda:0'):
    if isinstance(data, torch.Tensor): 
        return data.to(device, non_blocking=True)
    elif isinstance(data, dict):
        for k in data.keys():
            data[k] = to_device(data[k], device)

        return data
    elif isinstance(data, (tuple, list)):
        for i in range(len(data)):
            data[i] = to_device(data[i], device)

        return data
    
    return data

def set_requires_grad(model, value):
    for p in model.parameters():
        p.requires_grad = value

def freeze(model, b):
    set_requires_grad(model, not b)

def save_model(save_path, model, args=None, compress=False):
    model = unwrap_model(model)

    if not isinstance(args, dict):
        args = vars(args)

    dict_to_save = { 
        'state_dict': model.state_dict(),
        'args': args
    }

    if compress:
        with gzip.open(f'{save_path}.gz', 'wb') as f:
            torch.save(dict_to_save, f, pickle_protocol=-1)
    else:
        torch.save(dict_to_save, save_path, pickle_protocol=-1)
        
def load_model_checkpoint(path, model):
    if os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu')  
        model.load_state_dict(ckpt['state_dict'],strict=True)
    else:
        raise FileExistsError
    return model

def unwrap_model(model):
    model_ = model
    while True: 
        if isinstance(model_, torch.nn.DataParallel):
            model_ = model_.module
        elif isinstance(model_, ModelAndLoss):
            model_ = model_.model
        else:
            return model_


def to_numpy(t, flipy=False, uint8=True, i=0):
    out = t[:]
    if len(out.shape) == 4: # batched
        out = out[i]
    out = out.detach().permute(1, 2, 0) # HWC
    out = out.flip([0]) if flipy else out
    out = out.detach().cpu().numpy()
    out = (out.clip(0, 1)*255).astype(np.uint8) if uint8 else out
    return out


def get_module(path):
    m = pydoc.locate(path)
    assert m is not None, f'{path} not found'
    return m
