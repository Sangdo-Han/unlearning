from copy import deepcopy
from itertools import cycle

import numpy as np
import torch
from torch import nn, optim
from torch import functional as F

def srfl(
    net, 
    retain_loader, 
    forget_loader, 
    val_loader,
    class_weights=None,
    is_starter=False,
    **kwargs):
    """Simple unlearning by finetuning."""
    epochs = kwargs.get("epochs", 1)
    fweight = kwargs.get("fweight",0.8)
    flr = kwargs.get("lr", 0.001)
    DEVICE = kwargs.get("DEVICE", "gpu")
    temperature = kwargs.get("temperature", 3.0)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    rcriterion = nn.KLDivLoss(reduction="batchmean")
    fcriterion = nn.KLDivLoss(reduction="batchmean")

#   foptimizer = optim.Adam(net.parameters(), lr=flr)
    foptimizer = optim.SGD(net.parameters(), lr=flr,
                           momentum=kwargs.get("momentum", 0.9), weight_decay=kwargs.get("weight_decay", 1e-4))

    fscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        foptimizer, T_max=epochs)

    tnet = deepcopy(net)
    tnet.to("cuda")
    tnet.eval()

    for name, param in net.named_parameters():
        param.requires_grad = False
    for name, param in net.named_parameters():
        if name.count("layer1") or name.count("layer2"):
            param.requires_grad = True

    
    loss_record = []
    forget_loss_record= []
    retain_acc_record = []
    forget_acc_record = []
    for ep in range(epochs):
        flen = len(forget_loader)
        for idx, (retain_sample, forget_sample) in enumerate(zip(retain_loader, cycle(forget_loader))):
            if is_starter:
                retain_input, retain_targets = retain_sample
                forget_input, forget_targets = forget_sample
            else:
                retain_input = retain_sample["image"]
                retain_targets = retain_sample["age_group"]
                forget_input = forget_sample["image"]
                forget_targets = forget_sample["age_group"]

            retain_input, retain_targets = retain_input.to(DEVICE), retain_targets.to(DEVICE)
            forget_input, forget_targets = forget_input.to(DEVICE), forget_targets.to(DEVICE)

            with torch.no_grad():
                t_outputs = tnet(forget_input)
                ran_strides = np.random.choice([-3,3])
                pseudo_flabels = torch.roll(t_outputs, ran_strides,1)
                pseudo_flabels = F.softmax(pseudo_flabels, dim=1)

                # retain sampling
                retain_labels = tnet(retain_input)
                retain_labels = F.softmax(retain_labels, dim=1)
                
                # regularization - cosine similarity
                t_out_normed = F.normalize(t_outputs, p=2, dim=1)

            forget_outputs = net(forget_input)
            forget_out_normed = F.normalize(forget_outputs, p=2, dim=1)

            forget_outputs = F.log_softmax(forget_outputs, dim=1)
            
            # cosine similarity for regularization
            cosine_sim = 1.0 - F.cosine_similarity(forget_out_normed, t_out_normed, dim=1).mean()
            cosine_sim = - cosine_sim
            
            retain_outputs = net(retain_input)
            retain_log_outputs = F.log_softmax(retain_outputs, dim=1)

            _forget_loss =  0.5 * (1-fweight) * rcriterion(retain_log_outputs, retain_labels)\
                        + fweight * fcriterion(forget_outputs, pseudo_flabels) \
                        + 0.5 * (1-fweight) * criterion(retain_outputs, retain_targets)

            _retain_loss = criterion(retain_outputs, retain_targets)

            foptimizer.zero_grad()
            _forget_loss.backward(retain_graph=True)
            
            _retain_loss.backward(retain_graph=True)

            cosine_sim.backward()
            foptimizer.step()


            loss_record.append(_retain_loss.detach().cpu().numpy())
            retain_acc_record.append(torch.sum(torch.max(retain_outputs,1)[1] == retain_targets)\
                                     .detach().cpu().numpy()\
                                        / len(retain_outputs))
            forget_loss_record.append(_forget_loss.detach().cpu().numpy())
            forget_acc_record.append(torch.sum(torch.max(forget_outputs,1)[1] == forget_targets)\
                                     .detach().cpu().numpy()\
                                        / len(forget_outputs))

        fscheduler.step()

    net.eval()
    return loss_record, forget_loss_record, retain_acc_record, forget_acc_record