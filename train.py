import json
import argparse
import os
import time
import torch

from seger.models.networks import get_network
from trainer import SemanticTrainer

parser = argparse.ArgumentParser(description="Image to mask work")
parser.add_argument("-cfg", "--cfg_loc", default="cfgs/baseline.json")

args = parser.parse_args()

with open(args.cfg_loc, "r") as fp:
    cfg = json.load(fp)

print(cfg)
        
print("Model loader ....")
model = get_network(cfg["data"]["input_channels"], cfg["data"]["num_classes"], cfg["arch"]["arch_name"], cfg["arch"]["arch_kwargs"])

if cfg["train"]["use_pretrained_weights_loc"] is not None:
    print("Loading pretrained weights")
    save = torch.load(cfg["train"]["use_pretrained_weights_loc"])
    model.load_state_dict(save["model_weights"])


print("Initializing the trainer......")
model_trainer = SemanticTrainer(cfg, model)

best_loss = float("Inf")

if not os.path.exists("Experiments/" + cfg["model_stuff"]["exp_name"]):
    os.makedirs("Experiments/" + cfg["model_stuff"]["exp_name"])

if cfg["train"]["resume_training"] is not None:
    print("resuming training")
    best_loss = model_trainer.load_model()
    print("best_previous_loss: {}".format(best_loss))


overall_clock = time.clock()

logger = open("Experiments/" + cfg["model_stuff"]["exp_name"] + "/training_log.csv","w")

for ep in range(model_trainer.epochs, cfg["train"]["epochs"]):
    start = time.clock()
    print("[Epoch {}]".format(ep))
    train_loss,train_accuracy = model_trainer.trainer(ep)
    val_accuracy,val_loss = model_trainer.validate(ep)
    
    if val_loss<best_loss:
        best_loss = val_loss
        save_loc = model_trainer.save_model(val_loss)

    model_trainer.epochs = model_trainer.epochs + 1
    
    logger.write("{},{},{},{},{}".format(ep,train_loss,val_loss,train_accuracy,val_accuracy))
    logger.write("\n")
    
    print("train_loss: {} time_taken: {}, saved_at: {}".format(train_loss, 
                                                                time.clock() - start, 
                                                                save_loc))
print("Training completed")
print("total_training_time: {}".format(time.clock()- start))
