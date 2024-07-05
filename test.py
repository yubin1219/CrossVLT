import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

import transforms as T
import utils
from CrossVLT import SegModel

import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.utils as vutils
import random
import matplotlib.pyplot as plt
def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes

def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
    """
    Use for visualiation of segmentation results
    """
    from scipy.ndimage import binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)

def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7,.8,.9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'
    iters = 0
    
    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions, sent_list, img_ndarray = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            # gt = torch.cat([target,target,target], dim=0)
            #original_h, original_w = img_ndarray.size(1), img_ndarray.size(2)
                        
            # GT = F.interpolate(target.unsqueeze(0).float(), (original_h, original_w))
            # GT = GT.squeeze()
            # GT = GT.cpu().data.numpy()
            # GT = GT.astype(np.int8)
            target = target.cpu().data.numpy()
            # img_ndarray = img_ndarray.squeeze()
            # img_ndarray = img_ndarray.cpu().data.numpy()
            
            iters += 1
            for j in range(sentences.size(-1)):
                output  = model(image, sentences[:,:,j], attentions[:,:,j])                
                
                output_mask = output.argmax(1) # (1, 1, 480, 480)
                
                # result = output.argmax(1, keepdim=True)

                # result = F.interpolate(result.float(), (original_h, original_w))
                
                # result = result.squeeze()                

                # masks_pred = torch.cat([output_mask,output_mask,output_mask], dim=0)
                output_mask = output_mask.cpu().data.numpy()
                
                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1
                # sen = sent_list[j][0].replace('/','')
 
    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def main(args):
    device = torch.device('cuda')
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    if args.swin_type == "small":
        embed_dim=96
        num_heads=[3, 6, 12, 24]
        window_size=7
    elif args.swin_type == "base":
        embed_dim=128
        num_heads=[4, 8, 16, 32]
        window_size=12

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
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
                use_checkpoint=False,
                training=False
                )
    model.to(device)
    
    checkpoint = torch.load(args.resume, map_location=device)
    
    model.load_state_dict(checkpoint)
    
    evaluate(model, data_loader_test, device=device)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
