""" A trainer recipe to train Sementatic segmentation work
"""
import numpy as np
from collections import OrderedDict

import warnings

warnings.filterwarnings("ignore")

from seger.loss import get_loss
from seger.optims import get_optim, get_schedulers
from seger.datasets.seg_dataset import LaneDataset
from seger.vis_lib.visdom_vis import VisdomLinePlotter
from seger.utils import AverageMeter
from seger.metric import IOU,fscore_batch,get_metric
import torch


class SemanticTrainer():
    """Trains the network.

    cfg: config file
    model: Network architecture
    """

    def __init__(self, cfg, model):
        self.model = model
        self.cfg = cfg
        if self.cfg["device"]["use_gpu"]:
            self.model = self.model.cuda()
        if self.cfg["device"]["use_data_parallel"]:
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=self.cfg["device"]["device_ids"])
        loss_kwargs = {"num_classes": self.cfg["data"]["num_classes"], "weights": self.cfg["train"]["loss_kwargs"]["weights"]}
        
        self.criterion = get_loss(self.cfg["train"]["loss"], loss_kwargs)
        self.optimizer = get_optim(self.cfg["train"]["optim"],
                                   self.model,
                                   self.cfg["train"]["optim_kwargs"])
        self.scheduler = get_schedulers(self.cfg["train"]["sched_name"],
                                        self.optimizer,
                                        self.cfg["data"]["total_images"],
                                        self.cfg["train"]["batch_size"],
                                        self.cfg["train"]["sched_kwargs"])
        self.metric = get_metric(self.cfg["train"]["metric"])
        
        if self.cfg["device"]["use_visdom"]:
            self.plot = VisdomLinePlotter(env_name=self.cfg["model_stuff"]["exp_name"])
        self.epochs = 0
        

        self.trainset = LaneDataset(cfg, augmentations = True , train=True)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=cfg["train"]["batch_size"],
                                                       num_workers=2, shuffle=True)

        self.valset = LaneDataset(cfg, augmentations = False , train=False)
        self.valloader = torch.utils.data.DataLoader(self.valset, batch_size=cfg["train"]["batch_size"], num_workers=2,
                                                     shuffle=True)



    def trainer(self, ep):
        """ For each epoch, we need to train equally between all the img sizes,
        """

        self.model.train()
        total_iterations = len(self.trainloader) * ep
        loss_tracker = []
        accuracy_tracker = []
        self.model.zero_grad()
        loss_m, acc_m = AverageMeter(), AverageMeter()
        for num, k in enumerate(self.trainloader):
            with torch.set_grad_enabled(True):
                inputs, targets = k
                self.scheduler.optimizer.zero_grad()
                
                lr = self.scheduler.fix_lr()
                
                inputs, targets = k
                if self.cfg["device"]["use_gpu"]:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                
                outputs = self.model(inputs)
                
                
                
                if isinstance(outputs, tuple):
                    _, __ = outputs
                    outputs = _                   
                    
                
                final_loss = self.criterion(outputs, targets)
                acc = self.metric(targets,outputs)
                
    
                final_loss.backward()
                self.scheduler.optimizer.step()
            
            
                acc_m.update(acc)
                loss_m.update(final_loss.data.item())

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4)
                

                print("Iter: {}, loss:{}, iou:{}".format(total_iterations + num,
                                                         np.array(float(final_loss.cpu())),
                                                         np.array(float(acc))))

                if self.cfg["device"]["use_visdom"]:
                        self.plot.plot("final_loss_tracker", "training",
                                       total_iterations + num,
                                       np.mean(loss_tracker))
                        try:
                            self.plot.plot("lr", "training", total_iterations + num, np.array(lr))
                        except:
                            pass

                loss_tracker.append(float(final_loss.cpu()))
                accuracy_tracker.append(float(acc))

        loss_tracker = np.mean(loss_tracker)
        accuracy_tracker = np.mean(accuracy_tracker)

        if self.cfg["device"]["use_visdom"] and self.epochs>=2:
            self.plot.plot("total_loss_epoch_level", "training", self.epochs, np.array(float(loss_tracker)))
            self.plot.plot("total_accuracy_epoch_level", "training", self.epochs, np.array(float(accuracy_tracker)))
            
        

        return loss_tracker,accuracy_tracker


    def validate(self,ep):
        """ For each epoch, we need to train equally between all the img sizes,
        """
        
        self.model.eval()
        total_iterations = len(self.valloader) * ep

        loss_m, acc_m = AverageMeter(), AverageMeter()
        loss_tracker,accuracy_tracker = list(),list()

        for num, k in enumerate(self.valloader):
            with torch.set_grad_enabled(False):

                self.model.zero_grad()
                inputs, targets = k
                

                if self.cfg["device"]["use_gpu"]:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                outputs = self.model(inputs)
                
                if isinstance(outputs, tuple):
                    _, __ = outputs
                    outputs = _  

                final_loss = self.criterion(outputs, targets)
                acc = self.metric(targets,outputs)

                acc_m.update(acc)
                loss_m.update(final_loss.data.item())


        print("val_loss after {} epochs is ".format(self.epochs), loss_m.avg, flush=True)
        print("val_acc after {} epochs is ".format(self.epochs), acc_m.avg, flush=True)



        if self.cfg["device"]["use_visdom"] and self.epochs>=2:
            self.plot.plot("total_loss_epoch_level", "validation", self.epochs, np.array(float(loss_m.avg)))
            self.plot.plot("total_accuracy_epoch_level", "validation", self.epochs, np.array(float(acc_m.avg)))
            
        return acc_m.avg,loss_m.avg
    
    def save_model(self, score):
        """ Saves a trained model at the call.
        """
        save = {}
        save["model_weights"] = self.model.state_dict()
        save["epochs"] = self.epochs
        save["scheduler"] = self.scheduler.state_dict()
        save["best_score"] = score
#         name = self.cfg["model_stuff"]["exp_name"] + "/" + "_" + str(self.epochs) + \
#                "_" + str(score) + ".pth"
        name = "Experiments/" + self.cfg["model_stuff"]["exp_name"] + "/" + "best.pth".format(self.epochs)
        torch.save(save, name)
        return name

    def load_model(self):
        """Loads pretrained weights and optimizer weights to begin the training
        """
        save = torch.load(self.cfg["train"]["resume_training"])
        if self.cfg["device"]["use_gpu"]:
            self.model.load_state_dict(save["model_weights"])
        else:
            self.model.load_state_dict(OrderedDict((k.split(".", 1)[1], v) \
                                                   for k, v in save["model_weights"].items()))
        self.scheduler.load_state_dict(save["scheduler"])
        self.epochs = save["epochs"]
        return save["best_score"]


