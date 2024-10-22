import cv2
import copy
import json
import math
import os
import sys
import time
from typing import Iterable

import numpy as np
from shapely.geometry import Polygon
import torch

import util.misc as utils


#from s3d_floorplan_eval.Evaluator.Evaluator import Evaluator
#from s3d_floorplan_eval.options import MCSSOptions
#from s3d_floorplan_eval.DataRW.S3DRW import S3DRW
#from s3d_floorplan_eval.DataRW.wrong_annotatios import wrong_s3d_annotations_list

wrong_s3d_annotations_list = [3261, 3271, 3276, 3296, 3342, 3387, 3398, 3466, 3496]

##from scenecad_eval.Evaluator import Evaluator_S3D_ext

from util.poly_ops import pad_gt_polys
from util.plot_utils import plot_room_map, plot_score_map, plot_floorplan_with_regions, plot_semantic_rich_floorplan

#options = MCSSOptions()
#opts = options.parse()####################FIXME - hardcoded values

#import matplotlib.pyplot as plt
#import trimesh

###NEW
from nadirfloor_evaluator import NadirFloor_Evaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for batched_inputs in metric_logger.log_every(data_loader, print_freq, header):
        ####TO DO samples->contour outside the network BEFORE moving it to gpu
        gt_instances = [x["instances"].to(device) for x in batched_inputs]
        ##print('DEBUG samples', samples[0].device)
        room_targets = pad_gt_polys(gt_instances, model.num_queries_per_poly, device)
        
        samples = [x["image"].to(device) for x in batched_inputs]
        
        outputs = model(samples)
                        
        loss_dict = criterion(outputs, room_targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict.items()}
        loss_dict_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_scaled, **loss_dict_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, data_loader, device): #####NB. used diring training
    #
    model.eval()
    criterion.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for batched_inputs in metric_logger.log_every(data_loader, 10, header):

        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"]for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]
        room_targets = pad_gt_polys(gt_instances, model.num_queries_per_poly, device)

        ####inference
        outputs = model(samples)

        loss_dict = criterion(outputs, room_targets)
        weight_dict = criterion.weight_dict

        bs = outputs['pred_logits'].shape[0]
        pred_logits = outputs['pred_logits']
        pred_corners = outputs['pred_coords']
        fg_mask = torch.sigmoid(pred_logits) > 0.5 # select valid corners
       

        # process per scene
        for i in range(bs):

             #####NEW - generic s3d format evaluator
            gt_polys = []
            for j, poly in enumerate(gt_instances[i].gt_masks.polygons):
                ##print('DEBUG poly',j)
                gt_polys.append(np.array(poly).reshape(-1,2).astype(np.int32))#####WRONG! here the scene is multi-room
                                
            evaluator = NadirFloor_Evaluator()##Evaluator_S3D_ext()
                                       
            print("Running Evaluation for scene %s" % scene_ids[i])

            fg_mask_per_scene = fg_mask[i]
            pred_corners_per_scene = pred_corners[i]

            room_polys = []
            
            # semantic_rich = 'pred_room_logits' in outputs
            # if semantic_rich:
            #     room_types = []
            #     window_doors = []
            #     window_doors_types = []
            #     pred_room_label_per_scene = pred_room_label[i].cpu().numpy()

            # process per room
            for j in range(fg_mask_per_scene.shape[0]):
                fg_mask_per_room = fg_mask_per_scene[j]
                pred_corners_per_room = pred_corners_per_scene[j]
                valid_corners_per_room = pred_corners_per_room[fg_mask_per_room]
                
                if len(valid_corners_per_room)>0:
                    corners = (valid_corners_per_room * 255).cpu().numpy()
                    corners = np.around(corners).astype(np.int32)

                    ##if not semantic_rich:
                    # only regular rooms
                    if len(corners)>=4 and Polygon(corners).area >= 100:
                            room_polys.append(corners)
                    # else:
                    #     # regular rooms
                    #     if pred_room_label_per_scene[j] not in [16,17]:
                    #         if len(corners)>=4 and Polygon(corners).area >= 100:
                    #             room_polys.append(corners)
                    #             room_types.append(pred_room_label_per_scene[j])
                    #     # window / door
                    #     elif len(corners)==2:
                    #         window_doors.append(corners)
                    #         window_doors_types.append(pred_room_label_per_scene[j])
                  
                        
            quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys, gt_polys=gt_polys)


            if 'room_iou' in quant_result_dict_scene:
                metric_logger.update(room_iou=quant_result_dict_scene['room_iou'])
            
            metric_logger.update(room_prec=quant_result_dict_scene['room_prec'])
            metric_logger.update(room_rec=quant_result_dict_scene['room_rec'])
            metric_logger.update(corner_prec=quant_result_dict_scene['corner_prec'])
            metric_logger.update(corner_rec=quant_result_dict_scene['corner_rec'])
            metric_logger.update(angles_prec=quant_result_dict_scene['angles_prec'])
            metric_logger.update(angles_rec=quant_result_dict_scene['angles_rec'])

            # if semantic_rich:
            #     metric_logger.update(room_sem_prec=quant_result_dict_scene['room_sem_prec'])
            #     metric_logger.update(room_sem_rec=quant_result_dict_scene['room_sem_rec'])
            #     metric_logger.update(window_door_prec=quant_result_dict_scene['window_door_prec'])
            #     metric_logger.update(window_door_rec=quant_result_dict_scene['window_door_rec'])

        loss_dict_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict.items() if k in weight_dict}
        loss_dict_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict.items()}
        metric_logger.update(loss=sum(loss_dict_scaled.values()),
                             **loss_dict_scaled,
                             **loss_dict_unscaled)

    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats

