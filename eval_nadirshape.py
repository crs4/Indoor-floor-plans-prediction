import os
import glob
##import json
import argparse
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
import math
import torch
import time

##import numpy.ma as ma
import cv2

###only for debug
import matplotlib.pyplot as plt
import sys
from thop import profile, clever_format


from nadirshape.dataset_sv import ZIND_S3D_Dataset, x2image ####gatedslinenet
from nadirshape.nadirshapenet import NadirShapeNet, load_trained_model



def_pth ='./nadirshape/ckpt/DEMO_RUNS/s3d_combo_depth_layout_floor_mhsa_lgt_loss_masked/best_valid.pth' ###

def_output_dir = 'results/'

def_img = '' 



def get_segmentation_masks(seg_pred):
    soft_sem = torch.softmax(seg_pred, dim = 1) #####TO DO - here semantic is given by clutter mask
    soft_sem = torch.argmax(soft_sem, dim=1, keepdim=True)
    soft_sem = torch.clamp(soft_sem, min=0, max=1)
    masks = torch.zeros_like(seg_pred).to(seg_pred.device)
    masks.scatter_(1, soft_sem, 1)
            
    return masks

def get_IoU(outputs : torch.Tensor, labels : torch.Tensor, EPS = 1e-6):
    outputs = outputs.int()
    labels = labels.int()
    # Taken from: https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    intersection = (outputs & labels).float().sum((-1, -2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((-1, -2))  # Will be zero if both are 0

    iou = (intersection + EPS) / (union + EPS)  # We smooth our devision to avoid 0/0

    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    # return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch
    return iou.mean(dim = (-1,-2)).sum()

def depth_metrics(input_gt_depth_image,pred_depth_image, verbose=True, get_log = True):
    ###STEP 0 #######################################################
    input_gt_depth = input_gt_depth_image.copy()
    pred_depth = pred_depth_image.copy()

    n = np.sum(input_gt_depth > 1e-3) ####valid gt pixels count             
    idxs = ( (input_gt_depth <= 1e-3) ) ####valid gt pixels indices
    
    pred_depth[idxs] = 1 ### mask to 1 invalid pixels
    input_gt_depth[idxs] = 1 ### mask to 1 invalid pixels   

    print('valid samples:',n,'masked samples:', np.sum(idxs))

    ####STEP 1: compute delta######### FCRN standard
    pred_d_gt = pred_depth / input_gt_depth
    pred_d_gt[idxs] = 100
    gt_d_pred = input_gt_depth / pred_depth
    gt_d_pred[idxs] = 100

    Threshold_1_25 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25) / n
    Threshold_1_25_2 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25) / n
    Threshold_1_25_3 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25 * 1.25) / n
    ########################################################################################        

    #####STEP 2: compute mean errors################ OmniDepth,HoHoNet, etc. standard
    
    ##normalize to ground truth max

    input_gt_depth_norm = input_gt_depth / np.max(input_gt_depth)
    pred_depth_norm = pred_depth / np.max(input_gt_depth)     
        
        
    ARD = np.sum(np.abs((pred_depth_norm - input_gt_depth_norm)) / (input_gt_depth_norm) / n)
    SRD = np.sum(((pred_depth_norm - input_gt_depth_norm)** 2) / (input_gt_depth_norm) / n)

    ###
    log_pred_norm = np.log(pred_depth_norm)
    log_gt_norm = np.log(input_gt_depth_norm)
       
    RMSE_linear = np.sqrt(np.sum((pred_depth_norm - input_gt_depth_norm) ** 2) / n) ####FIXME - original without norm in any case
    
    if(get_log):
        RMSE_log = np.sqrt(np.sum((log_pred_norm - log_gt_norm) ** 2) / n)   ####FIXME - original without norm in any case 
    else:
        RMSE_log = 0.0
      

    if(verbose):
        print('Threshold_1_25: {}'.format(Threshold_1_25))
        print('Threshold_1_25_2: {}'.format(Threshold_1_25_2))
        print('Threshold_1_25_3: {}'.format(Threshold_1_25_3))
        print('RMSE_linear: {}'.format(RMSE_linear))
        print('RMSE_log: {}'.format(RMSE_log))
        print('SRD (MSE): {}'.format(SRD))
        print('ARD (MAE): {}'.format(ARD))
        
    return Threshold_1_25,Threshold_1_25_2,Threshold_1_25_3, RMSE_linear,RMSE_log,ARD,SRD

def inference(net, x, device):     
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
        
    start.record()
    depth, mask = net(x.to(device))
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print('time cost',start.elapsed_time(end))
                    
    return depth.cpu(),mask.cpu()



def test_nadirshape():
    ##
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', required=False, default = def_pth,
                        help='path to load saved checkpoint.')
    parser.add_argument('--root_dir', required=False, default = './data/s3d_single/test/')
    parser.add_argument('--output_dir', required=False, default = def_output_dir)
    parser.add_argument('--visualize', action='store_true', default = True)
    parser.add_argument('--no_cuda', action='store_true', default = False)
    parser.add_argument('--data_type', required=False, default = 's3d')    
    
    args = parser.parse_args()
        
    # Check target directory
    if not os.path.isdir(args.output_dir):
        print('Output directory %s not existed. Create one.' % args.output_dir)
        os.makedirs(args.output_dir)

    device = torch.device('cpu' if args.no_cuda else 'cuda')

    # Loaded trained model
    net = load_trained_model(NadirShapeNet, args.pth).to(device)
    net.eval()

    ####DEBUG comp stats
    get_comp_stats = True

    if(get_comp_stats):
        inputs = []
        img = torch.randn(1, 3, 512, 1024).to(device)
        inputs.append(img)
        
        with torch.no_grad():
            flops, params = profile(net, inputs)
            ##print(f'input :', [v.shape for v in inputs])
            print(f'flops : {flops/(10**9):.2f} G')
            print(f'params: {params/(10**6):.2f} M')

            import time
            fps = []
            with torch.no_grad():
                out,_ = net(img)
                      
        for _ in range(50):
            eps_time = time.time()
            net(img)
            torch.cuda.synchronize()
            eps_time = time.time() - eps_time
            fps.append(eps_time)
        print(f'fps   : {1 / (sum(fps) / len(fps)):.2f}')            
        


    # Inferencing
    
    evaluation_mode = False
                
    if(evaluation_mode):
        num_samples = 0
               
        dataset = ZIND_S3D_Dataset(root_dir=args.root_dir, return_name=True, full_size_rgb=True, get_depth = True,
                          get_layout = False, get_atl_layout = True, use_ceiling = False, get_max_min = True, get_depth_strip=True,
                          data_type=args.data_type)

        
        valid_count = 0
        outls = 0  
        
        iou2D = 0.0
        iou3D = 0.0

        Threshold_1_25 = 0
        Threshold_1_25_2 = 0
        Threshold_1_25_3 = 0
        RMSE_linear = 0.0
        RMSE_log = 0.0  
        RMSE_log_scale_invariant = 0.0
        ARD = 0.0
        SRD = 0.0
        SIM = 0.0

        N = len(dataset)

        for idx in range(N):
            
            x_img, x_depth, x_depth_s, x_atl_layout, max_min, f_name = dataset[idx]
            

            with torch.no_grad():
                x_img = x_img.unsqueeze(0)
                
                src_depth = x_depth.unsqueeze(0).to(device) ####To shape 1xhxw
            
                x_depth_s = x_depth_s.unsqueeze(0).to(device) ####To shape 1xw

            
                depth, mask_pred = inference(net, x_img, device)##, x_edges) ####x_clutter_mask.unsqueeze(1), x_masked not used here

                                
                seg_masks = get_segmentation_masks(mask_pred)

                layout_mask_pred = seg_masks[:,:1]##
                              
                iou2d = get_IoU(layout_mask_pred, x_atl_layout.unsqueeze(0))

                max = max_min[0]
                min = max_min[1]

                max = max.numpy()
                min = min.numpy()
                                
                p_max, p_min = dataset.d2l.max_min_depth(depth)

                p_max = p_max.numpy()
                p_min = p_min.numpy()
                                
                h_pred_ratio = p_max / p_min
                h_gt_ratio = max / min

                iouhr = 1.0 - (np.abs(h_pred_ratio-h_gt_ratio)/h_gt_ratio)
                                
                iou3d = iou2d*iouhr
                
                T1, T2,T3, Rlin,Rlog,A,S = depth_metrics(x_depth.numpy().astype(np.float32), depth.squeeze(0).numpy().astype(np.float32), verbose = False)
                
                cond = iou2d > 0.1 #### filter outliers (SEE github instructions)                

                if(cond):#####FIXME
                    iou2D += iou2d
                    iou3D += iou3d
                    valid_count += 1

                    Threshold_1_25 += T1
                    Threshold_1_25_2 += T2
                    Threshold_1_25_3 += T3
                    RMSE_linear += Rlin
                    RMSE_log += Rlog
                    
                    ARD += A
                    SRD += S
                    
                else:
                    outls += 1

                    # with open('./zind_combo_outliers_07_sub_eccv2024.txt', 'a') as out_file:
                    #     line = str(idx)+' '+str(iou2d)+'\n'
                    #     out_file.write(line)                    

        if(valid_count>1):

            print('average values at',N)

            iou2D /= float(valid_count)
            iou3D /= float(valid_count)

            Threshold_1_25 /= valid_count
            Threshold_1_25_2 /= valid_count
            Threshold_1_25_3 /= valid_count
            RMSE_linear /= valid_count
            RMSE_log /= valid_count
            ##RMSE_log_scale_invariant /= num_samples
            ARD /= valid_count
            SRD /= valid_count
            SIM /= valid_count
            
            print('Threshold_1_25: {}'.format(Threshold_1_25))
            print('Threshold_1_25_2: {}'.format(Threshold_1_25_2))
            print('Threshold_1_25_3: {}'.format(Threshold_1_25_3))
            print('RMSE_linear: {}'.format(RMSE_linear))
            print('RMSE_log: {}'.format(RMSE_log))
            ##print('RMSE_log_scale_invariant: {}'.format(RMSE_log_scale_invariant))
            print('ARD (MAE): {}'.format(ARD))
            print('SRD (MRE): {}'.format(SRD))

            print('iou2D', iou2D, 'iou3D', iou3D) 
            print('valid samples', float(valid_count)/float(len(dataset)), valid_count,len(dataset)) 
    else:
        idx = 0#29##27##246##259##361##361                
           
                        
        dataset = ZIND_S3D_Dataset(root_dir=args.root_dir, return_name=True, full_size_rgb=True, get_depth = True,
                           get_layout = False, get_atl_layout = True, use_ceiling = False, get_max_min = True, get_depth_strip=True,
                           data_type=args.data_type)


        x_img, x_depth, x_depth_s, x_atl_layout, max_min, f_name = dataset[idx]

        print(f_name)      
                            
        with torch.no_grad():
            x_img = x_img.unsqueeze(0)
                                    
            x_img_c = x2image( x_img.squeeze(0)) 

            src_depth = x_depth.unsqueeze(0).to(device) ####To shape 1xhxw
            
            x_depth_s = x_depth_s.unsqueeze(0).to(device) ####To shape 1xw
            
            depth, mask_pred = inference(net, x_img, device)##
            
            seg_masks = get_segmentation_masks(mask_pred)

            layout_mask_pred = seg_masks[:,:1]##
                                                           
            iouc = get_IoU(layout_mask_pred, x_atl_layout.unsqueeze(0))
            print('iou2D', iouc)

            depth_metrics(x_depth.numpy().astype(np.float32), depth.squeeze(0).numpy().astype(np.float32))
            
            
            max = max_min[0]
            min = max_min[1]

            max = max.numpy()
            min = min.numpy()

            print('max min GT', max , min)

            p_max, p_min = dataset.d2l.max_min_depth(depth)

            p_max2 = p_max.numpy()
            p_min2 = p_min.numpy()

            print('max min pred', p_max , p_min)  
               
            
            h_pred_ratio = p_max2 / p_min2
            h_gt_ratio = max / min

            iouH = 1.0 - (np.abs(h_pred_ratio-h_gt_ratio)/h_gt_ratio)

            iou3D = iouc*iouH

            print('iouH',iouH,'iou3D',iou3D)
                                                  
               
            plt.figure(1)
            plt.title('GT Nadir shape')
            plt.imshow(x_atl_layout) 

            plt.figure(5)
            plt.title('rgb input')
            plt.imshow(x_img_c) 
                       
            plt.figure(2)
            plt.title('predicted Nadir shape softmax')
            plt.imshow(layout_mask_pred.squeeze(1).squeeze(0)) 
                        
            plt.figure(23)
            plt.title('predicted Nadir shape')
            plt.imshow(mask_pred[:,:1].squeeze(1).squeeze(0)) 

            plt.figure(4)
            plt.title('predicted depth')
            plt.imshow(depth.squeeze(0)) 

            plt.figure(6)
            plt.title('GT depth')
            plt.imshow(x_depth.squeeze(0))    
                                                                                                   
            
            plt.show()
            
 
if __name__ == '__main__':
        
    test_nadirshape()

    
    
    
            
            
                      
            

   
