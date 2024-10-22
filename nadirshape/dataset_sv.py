from locale import normalize
import os
##from tkinter import W
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import interp2d
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn

import glob

#from nadirshape.misc.epc import EPC
#from nadirshape.misc.atlanta_transform import E2P

from nadirshape.misc.d2l import D2L
import torchvision.transforms.functional as tv


####map-style dataset for Torch dataLoader

class ZIND_S3D_Dataset(data.Dataset): ######legacy loader which recovers depth from layout (works with every annotated datasets - without real captured depth) 
    def __init__(self, root_dir,
                 flip=False, rotate=False, gamma=False, stretch=False,
                 return_name=False, full_size_rgb = True,
                 get_depth = True, get_depth_strip=False, get_mask_depth = False, get_src_edges = False, use_canny = True, get_layout = True, get_composed_depth = False,
                 get_atl_depth = False, get_atl_layout = False, get_max_min = False, W = 1024, H = 512, img_name_priority = False, use_ceiling = True, 
                 fp_fov = 165.0, data_type='zind'):
        
        ###source view
        self.img_dir = os.path.join(root_dir, 'img')
        ##self.depth_dir = os.path.join(root_dir, 'label')
        self.layout_dir = os.path.join(root_dir, 'label_cor')
        
        f_ext = '%s.jpg'
        
        if(data_type == 's3d'):
##print('data type', data_type)
            f_ext = '%s.png'

        if(img_name_priority):            
            self.img_fnames = sorted([
                fname for fname in os.listdir(self.img_dir)
                if fname.endswith('.jpg') or fname.endswith('.png')
            ])

            print('img count', len(self.img_fnames))

            self.txt_fnames = [f_ext % fname[:-4] for fname in self.img_fnames]
        else:            
            self.txt_fnames = sorted([
            fname for fname in os.listdir(self.layout_dir)
            if fname.endswith('.txt')
            ])            
        
            self.img_fnames = [f_ext % fname[:-4] for fname in self.txt_fnames]        
       
        ##self.depth_fnames = ['%s_depth.png' % fname[:-4] for fname in self.txt_fnames]                            
        
        self.get_layout = get_layout
                           
        self.full_size_rgb = full_size_rgb


        ####NEW 
        if(not self.full_size_rgb):
            W,H = 512, 256


        self.zind_camera_h = 1.7
        
        self.fp_fov = fp_fov#####default - is fine for zind

        self.d2l = D2L(gpu=False, H = H, W = W, fp_fov = self.fp_fov)

        self.get_atl_depth = get_atl_depth
        self.get_atl_layout = get_atl_layout
       
        self.get_max_min = get_max_min                                              

        self.metric_scale = 1000.0###default
        
        self.flip = flip
        self.rotate = rotate
        self.gamma = gamma
        
        self.return_name = return_name
                               
        self.get_depth = get_depth###NB. zind depth needs layout

        self.get_mask_depth = get_mask_depth
        
        self.get_depth_strip = get_depth_strip

        self.strip_h = 1
               
        ####edges
        self.get_src_edges = get_src_edges
        
        self.th_edges = 0.1

        self.use_canny = use_canny

        self.ed = None

        self.max = 0.0
        self.min = 0.0

        self.get_composed_depth = get_composed_depth

        self.use_ceiling = use_ceiling
                                

        if(self.get_src_edges):
            if(~self.use_canny):
                #    self.ed = Canny(threshold=1.0, use_cuda = False)
                #else:
                self.ed = EdgeDetector()
                                         
        self._check_dataset()


    def _check_dataset(self):
        for fname in self.img_fnames:
            #if(not os.path.isfile(os.path.join(self.depth_dir, fname))):
            #    print(fname)
            assert os.path.isfile(os.path.join(self.img_dir, fname)),\
                '%s not found' % os.path.join(self.img_dir, fname)

    def __len__(self):
        return len(self.img_fnames)
        

    def edgemap(self, img, mask = None):
        if(self.use_canny):
            ##blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = self.ed(img.unsqueeze(0))
            ##edge_im = thresholded##thin_edges##grad_mag
            ig = rgb_to_grayscale(img.unsqueeze(0), num_output_channels= 1)

            ig = ig.squeeze(0).squeeze(0).numpy().astype(np.float64)

            if(mask != None):
                mask = mask.squeeze(0).squeeze(0).numpy().astype(np.bool)

            edge_im = torch.FloatTensor(canny(ig, sigma=0.0, mask = mask)).unsqueeze(0).unsqueeze(0)

        else:
            ig = rgb_to_grayscale(img.unsqueeze(0), num_output_channels= 1)##rgb2gray(img)
            edge_im  = self.ed(ig)

        if(self.th_edges != -1):
            edge_im = torch.where(edge_im > self.th_edges, 1.0, 0.)

        
        return edge_im.squeeze(0) 
               

    def __getitem__(self, idx):
        # Read image
        img_path = os.path.join(self.img_dir,
                                self.img_fnames[idx])
                
        #depth_path = os.path.join(self.depth_dir,
        #                        self.depth_fnames[idx])
                                              

        if(self.full_size_rgb):
            img = np.array(Image.open(img_path), np.float32)[..., :3] / 255.
            
            full_H, full_W = img.shape[:2]
            H, W = full_H, full_W 
               
        else:
            img_pil = Image.open(img_path)            

            full_W,full_H = img_pil.size
            H, W = full_H//2,full_W//2
            
            img_pil = img_pil.resize((W,H), Image.BICUBIC)
            img = np.array(img_pil, np.float32)[..., :3] / 255.
            
        if(self.get_layout or self.get_atl_layout or self.get_depth):
            with open(os.path.join(self.layout_dir,
                               self.txt_fnames[idx])) as f:
                bon = np.array([line.strip().split() for line in f if line.strip()], np.float32)
                ##bon = cor###load_layout_from_txt(cor,H,W)###FIXME

                ##print('bon',bon)

                ###zind support from hn
                ##bon = np.roll(bon[:, :2], -2 * np.argmin(bon[::2, 0]), 0)
                ##c_ind = np.lexsort((bon[:,1],bon[:,0]))
                ##bon = bon[c_ind]

                ##print('bon',bon)

                if(not self.full_size_rgb):
                    bon /= 2
            #####NEW#### generate depth from layout
            depth = layout_2_depth(bon, H, W, min = self.zind_camera_h)


        
        # Random flip
        if self.flip and np.random.randint(2) == 0:
            img = np.flip(img, axis=1)

            ##self.flip_views(trg_img_list)
            
            if(self.get_depth):
                depth = np.flip(depth, axis=1)

            if(self.get_layout or self.get_atl_layout):
                bon = np.flip(bon, axis=1)                            
            
        # Random horizontal rotate
        if self.rotate:
            dx = np.random.randint(img.shape[1])
            img = np.roll(img, dx, axis=1)  
            
            ##trg_img_list = self.rotate_views(trg_img_list,dx)
            
            if(self.get_depth or self.get_atl_depth):
                depth = np.roll(depth, dx, axis=1)

            if(self.get_layout or self.get_atl_layout):
                bon = np.roll(bon, dx, axis=1)                           
            
        # Random gamma augmentation
        if self.gamma:
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 0:
                p = 1 / p
            img = img ** p 

            ##trg_img_list = self.gamma_views(trg_img_list,p)            

        scene_name = img_path[:-4]
        
        # Convert all data to tensor
        x = torch.FloatTensor(img.transpose([2, 0, 1]))##.copy())                                                       
               
        out_lst = [x]        
                
        if(self.get_depth):
            d = torch.FloatTensor(depth)##.copy())
            self.max, self.min = self.d2l.max_min_depth(d.unsqueeze(0))
            out_lst.append(d)   
            
        if(self.get_depth_strip):
            h,w = depth.shape
            ds = torch.FloatTensor(tv.center_crop(torch.FloatTensor(depth), (self.strip_h, w))).squeeze(0)
            out_lst.append(ds)
                                  
        
        if(self.get_mask_depth):
            dm = ((depth > 0.)).astype(np.uint8)  
                        
            m = torch.FloatTensor(dm.copy())
            out_lst.append(m)

        if(self.get_src_edges):
            em = self.edgemap(torch.FloatTensor(img.transpose([2, 0, 1])))
            ##e = torch.FloatTensor(em.copy())
            out_lst.append(em)

       
        if(self.get_layout):            
            out_lst.append(torch.FloatTensor(bon.copy()))


        if(self.get_composed_depth):####return src layout depth
            if(self.get_depth):
                H,W = torch.FloatTensor(depth).size()
                self.max, self.min = self.d2l.max_min_depth(torch.FloatTensor(depth).unsqueeze(0))
            l_depth = layout_2_depth(bon, H, W, min = self.min)#### hxw numpy

            depth_mask = ((depth > 0.)).astype(np.uint8)##(depth> 0), 1.0, 0.).float()

            l_mask = ((l_depth > 0.)).astype(np.uint8)

            valid_l = np.count_nonzero(l_mask)

            c_depth = depth

            if(valid_l>0):
                c_depth = depth_mask*depth + (1-depth_mask)*l_depth
            else:
                print('warning: invalid layout depth')

            out_lst.append(torch.FloatTensor(c_depth))

        
        if(self.get_atl_depth):
            d_up, d_down, c_dist, f_dist, fl = self.d2l.atlanta_transform_from_depth(torch.FloatTensor(depth).unsqueeze(0))
            if(self.use_ceiling):
                out_lst.append(torch.FloatTensor(d_up).squeeze(0))
            else:
                out_lst.append(torch.FloatTensor(d_down).squeeze(0))


        if(self.get_atl_layout):
            ###NB. needs  c_dist, f_dist or self.max, self.min
            mask, c_xy, self.max, self.min, tr_e_pts = self.d2l.atlanta_transform_from_equi_corners(bon, self.max, self.min, ceiling_mask = self.use_ceiling)
            out_lst.append(torch.FloatTensor(mask))

        if(self.get_max_min):
            max_min = np.zeros((2))
            max_min[0] = self.max
            max_min[1] = self.min
            out_lst.append(torch.DoubleTensor(max_min))

                    
        if self.return_name:
            out_lst.append(scene_name)

        return out_lst
######### 

def x2image(x):
    img = (x.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)

    return img

def layout_2_depth(cor_id, h, w, min, return_mask=False, get_depth_edges = False, filter_iter = 0):
    # Convert corners to per-column boundary first
    # Up -pi/2,  Down pi/2
            
    vc, vf = cor_2_1d(cor_id, h, w, to_angles = True)
                        
    vc = vc[None, :]  # [1, w]
    vf = vf[None, :]  # [1, w]
        
    ##FIXMEassert (vc > 0).sum() == 0
    ##FIXMEassert (vf < 0).sum() == 0

    # Per-pixel v coordinate (vertical angle)
    vs = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
    vs = np.repeat(vs[:, None], w, axis=1)  # [h, w]
        
    # Floor-plane to depth
    floor_h = min
    floor_d = np.abs(floor_h / np.sin(vs))

    ##print('layout h',floor_h)

    # wall to camera distance on horizontal plane at cross camera center
    cs = floor_h / np.tan(vf)

    
    # Ceiling-plane to depth
    ceil_h = np.abs(cs * np.tan(vc))      # [1, w]

    ##print('layout_2_depth h',floor_h, ceil_h)

    ceil_d = np.abs(ceil_h / np.sin(vs))  # [h, w]

    # Wall to depth
    wall_d = np.abs(cs / np.cos(vs))  # [h, w]

    # Recover layout depth
    floor_mask = (vs > vf)
    ceil_mask = (vs < vc)
    wall_mask = (~floor_mask) & (~ceil_mask)
       
    depth = np.zeros([h, w], np.float32)    # [h, w]
    depth[floor_mask] = floor_d[floor_mask]
    depth[ceil_mask] = ceil_d[ceil_mask]
    depth[wall_mask] = wall_d[wall_mask]

    ##assert (depth == 0).sum() == 0       
        

    if(get_depth_edges):
        vci, vfi = cor_2_1d(cor_id, h, w, to_angles = False)               
       
        vci = vci[None, :]  # [1, w]
        vfi = vfi[None, :]  # [1, w]

        ##vsi = np.arange(h)##((np.arange(h) + 0.5) / h - 0.5) * np.pi
        ##vsi = np.repeat(vsi[:, None], w, axis=1)  # [h, w]

        vx = np.arange(w)
        vx = vx[None, :]  # [1, w]

        ####cat coords

        vc_coords = np.concatenate((vx,vci),axis=0)
        vf_coords = np.concatenate((vx,vfi),axis=0)

        vc_coords = np.transpose(vc_coords,(1,0)).astype(int)
        vf_coords = np.transpose(vf_coords,(1,0)).astype(int)

        b_coords = np.concatenate((vc_coords, vf_coords),axis=0)
                
        cont_mask = np.zeros(shape=(h, w), dtype=np.uint8)
        
        ##print('b_coords',b_coords.shape,'cont_mask',cont_mask.shape)

        ##print(b_coords[:, 1], b_coords[:, 0])
        
        cont_mask[b_coords[:, 1], b_coords[:, 0]] = 1
               
        # Detect occlusion
        np_cor_id = cor_id

        ##print(np_cor_id)

        occlusion = find_occlusion(np_cor_id[::2].copy(), w = w, h = h).repeat(2)    
        np_cor_id = np_cor_id[~occlusion]

        ##print('occ',np_cor_id)
                
        ###TO DO draw vertical lines
        for i in range(len(np_cor_id)//2):
            p1 = np_cor_id[i*2].astype(int)
            p2 = np_cor_id[i*2+1].astype(int)

            x0 = p1[0]

            y1 = p1[1]
            y2 = p2[1]
                                            
            l = np.linspace(p1,p2,(y2-y1), retstep=True, dtype=int,axis=1)

            v_edge = np.transpose(l[0],(1,0)).astype(int)

            cont_mask[v_edge[:, 1], v_edge[:, 0]] = 1
       
        #plt.figure(456)
        #plt.title('DEBUG layout depth')
        #plt.imshow(cont_mask) 

        ## make edges in bold
        
        cont_mask = torch.FloatTensor(cont_mask).unsqueeze(0).unsqueeze(0)
        if(filter_iter>0):           
            #
            for i in range(filter_iter):        
                cont_mask = F.interpolate(cont_mask, size=(h//2, w//2), mode='bilinear', align_corners=False)
                cont_mask = F.interpolate(cont_mask, size=(h, w), mode='bilinear', align_corners=False)

        cont_mask = torch.where(cont_mask > 0.0, 1.0, 0.)
        cont_mask = cont_mask.squeeze(0).squeeze(0).numpy()


        return depth, cont_mask

    if return_mask:
        return depth, floor_mask, ceil_mask, wall_mask

    return depth

def cor_2_1d(cor, H, W, to_angles = False): #####return ceiling and floor boundaries as a 1D signal along W
    bon_ceil_x, bon_ceil_y = [], []
    bon_floor_x, bon_floor_y = [], []

    n_cor = len(cor)
    
    for i in range(n_cor // 2):
        xys = pano_connect_points(cor[i*2],
                                              cor[(i*2+2) % n_cor],
                                              z=-50, w=W, h=H)
        bon_ceil_x.extend(xys[:, 0])
        bon_ceil_y.extend(xys[:, 1])

    for i in range(n_cor // 2):
        xys = pano_connect_points(cor[i*2+1],
                                              cor[(i*2+3) % n_cor],
                                              z=50, w=W, h=H)
        bon_floor_x.extend(xys[:, 0])
        bon_floor_y.extend(xys[:, 1])

    bon_ceil_x, bon_ceil_y = sort_xy_filter_unique(bon_ceil_x, bon_ceil_y, y_small_first=True)
    bon_floor_x, bon_floor_y = sort_xy_filter_unique(bon_floor_x, bon_floor_y, y_small_first=False)

    bon = np.zeros((2, W))
    bon[0] = np.interp(np.arange(W), bon_ceil_x, bon_ceil_y, period=W)
    bon[1] = np.interp(np.arange(W), bon_floor_x, bon_floor_y, period=W)

    ##print(bon[1])

    if(to_angles):
        bon = ((bon + 0.5) / H - 0.5) * np.pi
        
    ###test
    #for j in range(len(bon[0])):
    #    now_x = bon[0, j]
    #    now_y = bon[1, j]
    #    print(now_x,now_y)

    return bon

def v2coory(v, h=512):
    return (v / np.pi + 0.5) * h - 0.5

def pano_connect_points(p1, p2, z=-50, w=1024, h=512):
    if p1[0] == p2[0]:
        return np.array([p1, p2], np.float32)

    u1 = coorx2u(p1[0], w)
    v1 = coory2v(p1[1], h)
    u2 = coorx2u(p2[0], w)
    v2 = coory2v(p2[1], h)

    x1, y1 = uv2xy(u1, v1, z)
    x2, y2 = uv2xy(u2, v2, z)

    if abs(p1[0] - p2[0]) < w / 2:
        pstart = np.ceil(min(p1[0], p2[0]))
        pend = np.floor(max(p1[0], p2[0]))
    else:
        pstart = np.ceil(max(p1[0], p2[0]))
        pend = np.floor(min(p1[0], p2[0]) + w)

    coorxs = (np.arange(pstart, pend + 1) % w).astype(np.float64)
    
    vx = x2 - x1
    vy = y2 - y1
    us = coorx2u(coorxs, w)
    ps = (np.tan(us) * x1 - y1) / (vy - np.tan(us) * vx)
    cs = np.sqrt((x1 + ps * vx) ** 2 + (y1 + ps * vy) ** 2)
    vs = np.arctan2(z, cs)

    coorys = v2coory(vs, h = h)
        
    return np.stack([coorxs, coorys], axis=-1) 

def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi


def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi

def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y

def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first)*2-1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys







    

    

   
   

            
    
