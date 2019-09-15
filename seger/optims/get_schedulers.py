""" Schedulars to fix learning_rate, weight_decay, momentum and other stuff after each iteration. 
"""

""" Schedulars to fix learning_rate, weight_decay, momentum and other stuff after each iteration. 
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 
import numpy as np

from functools import partial

def get_schedulers(sched_name, optimizer, train_images, batch_size, kwargs=None):
    if sched_name is None:
        print(sched_name, kwargs)
        if kwargs is None:
            kwargs = {"lr": 0.001}
        return No_scheduler(optimizer, 
                            **kwargs)

    elif sched_name == "sgdr":
        if kwargs is None:
            kwargs = {"t_actual": 2, "t_mul": 2, "lr_max": 0.05, "lr_min": 0.0, "decay": 1}
        
        return Warm_restarts(optimizer,
                     batch_size, 
                     train_images,
                     **kwargs)

    elif sched_name == "one_cycle":
        if kwargs is None:
            kwargs = {"stepsize": 4, "max_lr": 1, "base_lr": 0.1, "max_mom": 0.9, "base_mom": 0.9, "epochs": 10}
        
        return one_cycle(optimizer, 
                     batch_size, 
                     train_images,
                     **kwargs)

    elif sched_name == "cyclic_lr":
        if kwargs is None:
            kwargs = {"stepsize": 5, "max_lr": 0.1, "base_lr": 0.02}

        return cyclic_lr(optimizer,
                     batch_size,
                     train_images,
                     **kwargs)

    elif sched_name == "ReduceLRonPlateau":
        if kwargs is None:
            kwargs = {"mode": 'min', "factor": 0.2, "patience": 5, "verbose": False, "threshold": 1e-4, "threshold_mode": 'rel', 
                       "cooldown" : 0, "min_lr": 0.0, "eps": 1e-8}
        return ReduceLRonPlateau(optimizer,
                     batch_size, 
                     train_images,
                     **kwargs)

    elif sched_name == "StepLR":
        if kwargs is None:
            kwargs = {"step_size": 10, "gamma": 0.5, "last_epoch": -1}

        return StepLR(optimizer,
                     batch_size,
                     train_images,
                     **kwargs)
    elif sched_name == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80,100,120,140,160,180], gamma=0.9)
            
    
    else:
        raise NotImplementedError("Schedular is not implemented")


class No_scheduler:
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr
        self.cycle = 0 
        self.last_batch_iteration = 0 
    
    def get_lr(self):
        return self.lr
    
    def fix_lr(self, *argv):
        new_lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr 
    
    def load_state_dict(self, save):
        self.optimizer.load_state_dict(save["optimizer"])
        self.last_batch_iteration = save["last_batch_iteration"]
        self.cycle = save["cycle"]
    
    def state_dict(self):
        save = {}
        save["optimizer"] = self.optimizer.state_dict()
        save["last_best_iteration"] = self.last_batch_iteration 
        save["cycle"] = self.cycle 
        return save 

        
class Warm_restarts:
    def __init__(self, optimizer, batch_size, total_images, t_actual, t_mul, lr_max, lr_min, decay):
        """ SGDR warm restarts scheduler 
        optimizer: optimizer used for training 
        t_actual: 
        """
        self.optimizer = optimizer
        self.t_actual = t_actual
        self.t_mul = t_mul
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.decay = decay
        self.batch_size = batch_size 
        self.total_images = total_images
        self.last_batch_iteration = 0
        self.iterations_per_batch = math.ceil(self.total_images/self.batch_size)
        self.cycle = 0
        self.ti = self.iterations_per_batch * self.t_actual
    
    def get_lr(self):
        iteration = self.last_batch_iteration/self.ti
        cosine_value = math.cos(iteration * math.pi)
        new_lr = self.lr_min + (0.5 * (self.lr_max - self.lr_min) * (1+cosine_value))
        self.last_batch_iteration += 1
        
        if self.last_batch_iteration==self.ti:
            print("Warm restart")
            self.last_batch_iteration = 0
            self.ti *= self.t_mul
            self.cycle += 1 
            self.lr_max = self.lr_max/self.decay
        return new_lr
    
    def fix_lr(self, *argv):
        new_lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    
    def load_state_dict(self, save):
        self.optimizer.load_state_dict(save["optimizer"])
        self.last_batch_iteration = save["last_batch_iteration"]
        self.cycle = save["cycle"]
    
    def state_dict(self):
        save = {}
        save["optimizer"] = self.optimizer.state_dict()
        save["last_batch_iteration"] = self.last_batch_iteration 
        save["cycle"] = self.cycle 
        return save

class one_cycle():
    def __init__(self, optimizer, batch_size, total_images, stepsize, max_lr, base_lr, max_mom, base_mom, epochs):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.base_lr = base_lr
        self.max_mom = max_mom
        self.base_mom = base_mom
        self.onecycle = False
        self.batch_size = batch_size 
        self.total_images = total_images
        self.total_ep = epochs
        self.last_batch_iteration = 0
        self.iterations_per_batch = math.ceil(self.total_images/self.batch_size)
        self.stepsize = stepsize * self.iterations_per_batch
    
    def get_triangular_lr(self):
        cycle = np.floor(1 + self.last_batch_iteration/(2  * self.stepsize))
        x = np.abs(self.last_batch_iteration/self.stepsize - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1-x))
        return lr

    def get_inverted_triangular_mom(self):
        cycle = np.floor(1 + self.last_batch_iteration/(2  * self.stepsize))
        x = np.abs(self.last_batch_iteration/self.stepsize - 2 * cycle + 1)
        mom = self.max_mom + (self.base_mom - self.max_mom) * np.maximum(0, (1-x))
        return mom
    
    def fix_lr(self, *argv):
        new_lr = self.get_triangular_lr()
        new_mom = self.get_inverted_triangular_mom()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
            param_group["momentum"] = new_mom
        if (self.last_batch_iteration==2*self.stepsize):
            self.last_batch_iteration = 0
            self.stepsize = self.total_ep - (self.stepsize/self.iterations_per_batch)*2
            self.max_lr = 0
            self.base_mom = self.max_mom
        
        self.last_batch_iteration += 1
        return new_lr

    def load_state_dict(self, save):
        self.optimizer.load_state_dict(save["optimizer"])
        self.last_batch_iteration = save["last_batch_iteration"]
        self.stepsize = save["stepsize"]
    
    def state_dict(self):
        save = {}
        save["optimizer"] = self.optimizer.state_dict()
        save["last_batch_iteration"] = self.last_batch_iteration
        save["stepsize"] = self.stepsize
        return save


class cyclic_lr():
    def __init__(self, optimizer, batch_size, total_images, stepsize, max_lr, base_lr):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.base_lr = base_lr
        self.batch_size = batch_size 
        self.total_images = total_images
        self.last_batch_iteration = 0
        self.iterations_per_epoch = math.floor(self.total_images/self.batch_size)
        self.stepsize = stepsize * self.iterations_per_epoch
    
    def get_triangular_lr(self):
        cycle = np.floor(1 + self.last_batch_iteration/(2  * self.stepsize))
        x = np.abs(self.last_batch_iteration/self.stepsize - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1-x))
        self.last_batch_iteration += 1
        return lr
    
    def fix_lr(self, *argv):
        new_lr = self.get_triangular_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in {'optimizer'}}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)



class ReduceLRonPlateau():
    def __init__(self, optimizer, batch_size, total_images, mode, factor, patience,verbose, threshold, 
                 threshold_mode, cooldown, min_lr, eps):
        self.factor = factor
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self.batch_size = batch_size 
        self.total_images = total_images
        self.last_batch_iteration = -1
        self.iterations_per_batch = math.ceil(self.total_images/self.batch_size)
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def fix_lr(self, metrics, epoch=None):
        current = metrics
        self.last_batch_iteration += 1
        if (self.last_batch_iteration!=self.iterations_per_batch):
            return self.optimizer.param_groups[0]['lr']
        else:
            self.last_batch_iteration = 0
            if epoch is None:
                epoch = self.last_epoch = self.last_epoch + 1
            self.last_epoch = epoch

            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

            if self.num_bad_epochs > self.patience:
                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
            return self.optimizer.param_groups[0]['lr']

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = (-float('inf'))

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in {'optimizer', 'is_better'}}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)


class StepLR():
    def __init__(self, optimizer, batch_size, total_images, step_size, gamma, last_epoch):
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch
        self.step_size = step_size
        self.gamma = gamma
        self.batch_size = batch_size 
        self.total_images = total_images
        self.iterations_per_batch = math.ceil(self.total_images/self.batch_size)
        self.last_batch_iteration = self.iterations_per_batch - 1
        self.fix_lr(last_epoch + 1)
        self.last_batch_iteration = -1

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]

    def fix_lr(self, epoch=None):
        self.last_batch_iteration += 1
        if (self.last_batch_iteration!=self.iterations_per_batch):
            return self.optimizer.param_groups[0]['lr']
        self.last_batch_iteration = 0
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in {'optimizer'}}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        
        
    