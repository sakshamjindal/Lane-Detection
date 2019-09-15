import torch
from torchvision.transforms import transforms
import numpy as np
import visdom

import matplotlib.pyplot as plt 
from PIL import Image


class VisdomLinePlotter(object):
    """Plots to Visdom
    """
    def __init__(self, env_name='main'):
        self.viz = visdom.Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y):
        """ Plot all the metrics available as named tuple
        
        var_name: named tuple 
        """
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Iterations',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

    def plot_tuple(self, ntuple, name, iteration):

        for field, value in ntuple._asdict().items():
            self.plot(field, name, iteration, value.data.cpu().numpy()[0])


def vis_images(images_tensor, labels, imgs_per_row=4, mode="RGB", fig_size= (20, 3)):
    if isinstance(images_tensor, list):
        pil_images = [Image.open(i) for i in images_tensor]
    else:
        pil_convertor = transforms.ToPILImage(mode=mode)
        pil_images=[pil_convertor(i) for i in images_tensor]
    if labels is None:
        labels= ["No label" for i in range(len(pil_images))]
    else:
        labels = [i[0] for i in labels.numpy()]
    
    batches = len(pil_images)//imgs_per_row
    for i in range(batches):
        imgs = pil_images[i*imgs_per_row:(i+1)*imgs_per_row]
        lab = labels[i*imgs_per_row:(i+1)*imgs_per_row]
        fig, ax = plt.subplots(nrows=1, ncols=4, sharex="col", sharey="row", figsize=fig_size)
        for i, img in enumerate(imgs):
            ax[i].imshow(img)
            ax[i].set_title(str(lab[i]))