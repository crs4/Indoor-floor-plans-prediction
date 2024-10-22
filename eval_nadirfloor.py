import argparse
import datetime
import json
import random
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
#from models import build_model

###NEW
from util.plot_utils import plot_room_map, plot_score_map, plot_floorplan_with_regions
import cv2
from nadirfloor_evaluator import NadirFloor_Evaluator
from shapely.geometry import Polygon
import trimesh
import matplotlib.pyplot as plt
from datasets.poly_data import build as build_poly
from models.nadirfloornet import build_nadirfloor, get_floorplan

 

#####derived from RoomFormer evaluate; small adaption to parse unified s3d and zind data
@torch.no_grad()
def evaluate_nadirfloor(model, dataset_root, data_loader, device, output_dir, plot_pred=True, plot_density=True, plot_gt=True, 
                       eval_set = 'test', export_prediction = True, invalid_scenes_ids = None):
    ##model.eval()

    quant_result_dict = None
    scene_counter = 0
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
     
    for batched_inputs in data_loader:
        ###########BUILD dataset from args path (i.e. COCO annotations and images)
        samples = [x["image"].to(device) for x in batched_inputs] ######input density_map - default: a list with 1 element: 1 x H x W
        scene_ids = [x["image_id"] for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs] ###### usually 1 - batch

        evaluator = NadirFloor_Evaluator()#####init evaluator for nadir planes from s3d or ZInD

        ####prepare GT annotations
        for i in range(len(gt_instances)):####### gt_instances are always 1 for evaluation                                    
            gt_polys = []
                            
            for j, poly in enumerate(gt_instances[i].gt_masks.polygons):
                gt_polys.append(np.array(poly).reshape(-1,2).astype(np.int32))  

        metric_info = None                        
        
        if (export_prediction):
            metric_info = dataset_root+'/'+eval_set+'/'+f'{scene_ids[i]:05d}'+'.json'####NB. predicted by nadir shape module ###CHECK i                                 
                   
        ###logits and queries prediction
        outputs = model(samples)
         
        room_polys, floorplan3d = get_floorplan(outputs, metric_info)                                                       

        quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys, gt_polys=gt_polys)
                                                
            
        
        print('processing',str(scene_ids[i]))                                                        
        if quant_result_dict is None:
            quant_result_dict = quant_result_dict_scene
        else:
            for k in quant_result_dict.keys():
                quant_result_dict[k] += quant_result_dict_scene[k]

        scene_counter += 1

        # draw GT map
        if plot_gt:
            for i, gt_inst in enumerate(gt_instances): #####for each SCENE                       
                # plot regular room floorplan
                gt_polys = []
                density_map = np.transpose((samples[i] * 255).cpu().numpy(), [1, 2, 0])
                density_map = np.repeat(density_map, 3, axis=2)
                
                gt_corner_map = np.zeros([256, 256, 3])
                for j, poly in enumerate(gt_inst.gt_masks.polygons):
                    corners = poly[0].reshape(-1, 2) ####as corners
                    gt_polys.append(corners)
                        
                gt_room_polys = [np.array(r) for r in gt_polys]
                gt_floorplan_map = plot_floorplan_with_regions(gt_room_polys, scale=1000)
                                                                                
                cv2.imwrite(os.path.join(output_dir, '{}_gt.png'.format(scene_ids[i])), gt_floorplan_map)
                
        if plot_pred:
            # plot regular room floorplan
            room_polys = [np.array(r) for r in room_polys]
                        
            floorplan_map = plot_floorplan_with_regions(room_polys, scale=1000)
            cv2.imwrite(os.path.join(output_dir, '{}_pred_floorplan.png'.format(scene_ids[i])), floorplan_map)

        if plot_density:
            density_map = np.transpose((samples[i] * 255).cpu().numpy(), [1, 2, 0])
            density_map = np.repeat(density_map, 3, axis=2)
            pred_room_map = np.zeros([256, 256, 3])

            for room_poly in room_polys:
                pred_room_map = plot_room_map(room_poly, pred_room_map)

            # plot predicted polygon overlaid on the density map
            pred_room_map = np.clip(pred_room_map + density_map, 0, 255)
               
            cv2.imwrite(os.path.join(output_dir, '{}_pred_room_map.png'.format(scene_ids[i])), pred_room_map)
                
        if (export_prediction):
            print('export predicted polygons')                                 
                   
            obj_name = output_dir+'/'+f'{scene_ids[i]:05d}'+'.obj'   

            print('saving obj at', obj_name)                                                               
                    
            trimesh.exchange.export.export_scene(floorplan3d, obj_name)                      
                                 
    for k in quant_result_dict.keys():
        quant_result_dict[k] /= float(scene_counter)

    metric_category = ['room','corner','angles']
    for metric in metric_category:
        prec = quant_result_dict[metric+'_prec']
        rec = quant_result_dict[metric+'_rec']
        f1 = 2*prec*rec/(prec+rec)
        quant_result_dict[metric+'_f1'] = f1

    print("*************************************************")
    print(quant_result_dict)
    print("*************************************************")

    with open(os.path.join(output_dir, 'results.txt'), 'w') as file:
        file.write(json.dumps(quant_result_dict))

def get_args_parser():
    parser = argparse.ArgumentParser('OmniFloor', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)

    # backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--lr_backbone', default=0, type=float)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=800, type=int,
                        help="Number of query slots (num_polys * max. number of corner per poly)")
    parser.add_argument('--num_polys', default=20, type=int,
                        help="Number of maximum number of room polygons")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    parser.add_argument('--query_pos_type', default='sine', type=str, choices=('static', 'sine', 'none'),
                        help="Type of query pos in decoder - \
                        1. static: same setting with DETR and Deformable-DETR, the query_pos is the same for all layers \
                        2. sine: since embedding from reference points (so if references points update, query_pos also \
                        3. none: remove query_pos")
    parser.add_argument('--with_poly_refine', default=True, action='store_true',
                        help="iteratively refine reference points (i.e. positional part of polygon queries)")
    parser.add_argument('--masked_attn', default=False, action='store_true',####exp
                        help="if true, the query in one room will not be allowed to attend other room")
    parser.add_argument('--semantic_classes', default=-1, type=int,
                        help="Number of classes for semantically-rich floorplan:  \
                        1. default -1 means non-semantic floorplan \
                        2. 19 for Structured3D: 16 room types + 1 door + 1 window + 1 empty")

    # aux
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_true',
                        help="Disables auxiliary decoding losses (loss at each layer)")          

    # dataset parameters
    parser.add_argument('--dataset_root', default='./results/s3d_nadirmaps', type=str)###SUB BEST  
      
    
    parser.add_argument('--eval_set', default='test', type=str)

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
      
    ###model checkpoints  
    parser.add_argument('--checkpoint', default='checkpoints/DEMO_RUNS/nadirfloornet_s3d.pth', help='resume from checkpoint')##  ##
    parser.add_argument('--output_dir', default='eval_nadirfloor_s3d_demo',help='path where to save result')#####    
                                

    # visualization options
    parser.add_argument('--plot_pred', default=True, type=bool, help="plot predicted floorplan")
    parser.add_argument('--plot_density', default=True, type=bool, help="plot predicited room polygons overlaid on the density map")
    parser.add_argument('--plot_gt', default=True, type=bool, help="plot ground truth floorplan")

    parser.add_argument('--seg_image', default=False, type=bool, help='nadir map is a segmentation map')###NOT USED HERE - default: False
    parser.add_argument('--save_metric', default=False, type=bool, help='save metric information')#### NOT USED HERE - default: False
     
    parser.add_argument('--save_model', default=False, type=bool, help='save metric information')
        
    parser.add_argument('--seed', default=42, type=int)  ####NB. 42 same of RoomFormer for comparison 
        

    return parser

    
def main(args):
    
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)    

    device = torch.device(args.device)
    
    # build model
    model = build_nadirfloor(args, train=False)
    model.to(device)
      
    dataset_eval = build_poly(args.eval_set, args)############### BUILD dataset from args path (i.e. COCO annotations and images)
    sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)
       
    def trivial_batch_collator(batch):
        """
        A batch collator that does nothing.
        """
        return batch

    data_loader_eval = DataLoader(dataset_eval, args.batch_size, sampler=sampler_eval, drop_last=False, collate_fn=trivial_batch_collator, num_workers=0,pin_memory=True)
   
    output_dir = Path(args.output_dir)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]

    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))

    save_dir = os.path.join(os.path.dirname(args.checkpoint), output_dir)

    evaluate_nadirfloor(
                   model, args.dataset_root, data_loader_eval, 
                   device, save_dir, 
                   plot_pred=args.plot_pred, 
                   plot_density=args.plot_density, 
                   plot_gt=args.plot_gt,
                   export_prediction = args.save_model,                    
                   eval_set = args.eval_set, 
                   invalid_scenes_ids = []###
                   )

   

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
