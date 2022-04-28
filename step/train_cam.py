
import torch
import sys
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F

import importlib

import voc12.dataloader
from misc import pyutils, torchutils


def validate(model, data_loader):
    sys.stderr.write('validating ... ')

    val_loss_meter = pyutils.AverageMeter('loss1', 'loss2')

    model.eval()

    with torch.no_grad():
        for pack in data_loader:
            img = pack['img']

            label = pack['label'].cuda(non_blocking=True)

            x = model(img)
            loss1 = F.multilabel_soft_margin_loss(x, label)

            val_loss_meter.add({'loss1': loss1.item()})

    model.train()

    sys.stderr.write('loss: %.4f' % (val_loss_meter.pop('loss1')))
    sys.stderr.write('\n')

    return


def run(args):

    #Import ResNet50 classifcation model attributes
    model = getattr(importlib.import_module(args.cam_network), 'Net')()


    #Import training dataset (voc12)
    train_dataset = voc12.dataloader.VOC12ClassificationDataset(args.train_list, voc12_root=args.voc12_root,
                                                                resize_long=(320, 640), hor_flip=True,
                                                                crop_size=512, crop_method="random")
    
    #Load training dataset (voc12)
    train_data_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)


    #Calculating max step for the network.
    max_step = (len(train_dataset) // args.cam_batch_size) * args.cam_num_epoches

    #Import validation dataset (voc12)
    val_dataset = voc12.dataloader.VOC12ClassificationDataset(args.val_list, voc12_root=args.voc12_root,
                                                              crop_size=512)
    
    #Load validation dataset (voc12)
    val_data_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    #Set training parameters for the model
    param_groups = model.trainable_parameters()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
        {'params': param_groups[1], 'lr': 10*args.cam_learning_rate, 'weight_decay': args.cam_weight_decay},
    ], lr=args.cam_learning_rate, weight_decay=args.cam_weight_decay, max_step=max_step)

    #Train the model with the attributes loaded
    model = torch.nn.DataParallel(model).cuda()
    model.train()


    avg_meter = pyutils.AverageMeter()

    timer = pyutils.Timer()

    #Iterate over epochs
    for ep in range(args.cam_num_epoches):

        sys.stderr.write('Epoch %d/%d' % (ep+1, args.cam_num_epoches))
        sys.stderr.write('\n')

        for step, pack in enumerate(train_data_loader):
            
            #Load image and image label
            img = pack['img']
            label = pack['label'].cuda(non_blocking=True)

            #calculate loss
            x = model(img)
            loss = F.multilabel_soft_margin_loss(x, label)

            avg_meter.add({'loss1': loss.item()})

            #optimize model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Print progress
            if (optimizer.global_step-1) % 100 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                sys.stderr.write('step:%5d/%5d ' % (optimizer.global_step - 1, max_step))
                sys.stderr.write('loss:%.4f ' % (avg_meter.pop('loss1')))
                sys.stderr.write('imps:%.1f ' % ((step + 1) * args.cam_batch_size / timer.get_stage_elapsed()))
                sys.stderr.write('lr: %.4f ' % (optimizer.param_groups[0]['lr']))
                sys.stderr.write('etc:%s ' % (timer.str_estimated_complete()))
                sys.stderr.write('\n')

        else:
            validate(model, val_data_loader)
            timer.reset_stage()

    #Save the final model
    torch.save(model.module.state_dict(), args.cam_weights_name + '.pth')
    torch.cuda.empty_cache()