import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from functools import reduce
import operator

import torchvision

import transforms as T
import utils
import numpy as np
import random
import torch.nn.functional as F

import gc
from CrossVLT import SegModel
from data.dataset_refer_bert import ReferDataset

def get_dataset(image_set, transform, args, eval_mode=False):
    if eval_mode:
        ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    else:
        ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None
                      )
    num_classes = 2

    return ds, num_classes


# IoU calculation for validation
def IoU(pred, gt):
    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou, intersection, union


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def criterion(input, target):
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return nn.functional.cross_entropy(input, target, weight=weight).cuda()


def evaluate(model, data_loader):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions, _, _ = data
            image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                                   target.cuda(non_blocking=True),\
                                                   sentences.cuda(non_blocking=True),\
                                                   attentions.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            for j in range(sentences.size(-1)):
                output, sim1, sim2, sim3, sim4 = model(image, sentences[:,:,j], attentions[:,:,j])

                iou, I, U = IoU(output, target)
                acc_ious += iou
                mean_IoU.append(iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (iou >= eval_seg_iou)
                seg_total += 1
        iou = acc_ious / seg_total

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    return 100 * iou, 100 * cum_I / cum_U


def train_one_epoch(model, criterion, criterion2, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
        total_its += 1
        image, target, sentences, attentions = data
        image, target, sentences, attentions = image.cuda(non_blocking=True),\
                                               target.cuda(non_blocking=True),\
                                               sentences.cuda(non_blocking=True),\
                                               attentions.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        output, sim1, sim2, sim3, sim4 = model(image, sentences, attentions)
    
        loss = 2 * criterion(output, target.detach())
        
        loss2 = criterion2(sim1, target.detach()) +criterion2(sim2, target.detach()) + criterion2(sim3, target.detach()) + criterion2(sim4, target.detach())
        loss = loss + loss2

        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), loss2 = loss2.item(),  lr=optimizer.param_groups[0]["lr"])

        torch.cuda.synchronize()

class AlignLoss(nn.Module):
    def __init__(self):
        super(AlignLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')        
        
    def forward(self, m, target):
        x = F.interpolate(m, size=(480,480), mode='bilinear', align_corners=True)
        loss = self.loss(x.squeeze(1), target.float())
        
        return loss

def main(args):
    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args)
    dataset_test, _ = get_dataset("val",
                                  get_transform(args=args),
                                  args=args,
                                  eval_mode=True)
                                  

    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                                    shuffle=True)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    if args.swin_type == "small":
        embed_dim=96
        num_heads=[3, 6, 12, 24]
        window_size=7
    if args.swin_type == "base":
        embed_dim=128
        num_heads=[4, 8, 16, 32]
        window_size=12
    print("embed_dim : ",embed_dim)

    model = SegModel(args,
                pretrain_img_size=384,
                patch_size=4,
                embed_dim=embed_dim,
                depths=[2, 2, 18, 2],
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
                use_checkpoint=False,
                training=True
                ) 
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    print('Distributed model')
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    single_model = model.module

    criterion2 = AlignLoss().to('cuda')
    best_oIoU = 0.0
    
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)
    
    params_to_optimize = [
        {'params': backbone_no_decay, 'weight_decay': 0.0},
        {'params': backbone_decay},
        {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
        {"params": [p for p in single_model.lang_stage1.encoder.parameters() if p.requires_grad]},
        {"params": [p for p in single_model.lang_stage2.parameters() if p.requires_grad]},
        {"params": [p for p in single_model.lang_stage3.parameters() if p.requires_grad]},
        {"params": [p for p in single_model.lang_stage4.parameters() if p.requires_grad]}
    ]
    print('Optim & LR scheduler')
    optimizer = torch.optim.AdamW(params_to_optimize,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  amsgrad=args.amsgrad
                                  )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)
    
    start_time = time.time()
    iterations = 0
    print("Best oIoU : ", best_oIoU)
    resume_epoch = -999
    print('-----Start Training-----')
    for epoch in range(max(0, resume_epoch+1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        
        train_one_epoch(model, criterion, criterion2, optimizer, data_loader, lr_scheduler, epoch, args.print_freq,
                        iterations)

        iou, overallIoU = evaluate(model, data_loader_test)

        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))

        save_checkpoint = (best_oIoU < overallIoU)

        if save_checkpoint:
            best_oIoU = overallIoU
            best_epoch = epoch
            print('Better epoch: {}\n'.format(epoch))
            dict_to_save = {'model': single_model.state_dict(),
                             'epoch': epoch, 'args': args,
                            'best_oIoU':best_oIoU}
            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'refcoco_best.pth'))    
    
        else:
            dict_to_save = {'model': single_model.state_dict(),
                             'epoch': epoch, 'args': args,
                            'best_oIoU':best_oIoU}
            utils.save_on_master(dict_to_save, os.path.join(args.output_dir,
                                                            'refcoco_checkpoint.pth'))
     
        print("Best oIoU : ", best_oIoU)
        print("Best Epoch : ", best_epoch)
       
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    print("local rank = ",args.local_rank)
    utils.init_distributed_mode(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
