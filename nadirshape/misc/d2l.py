import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import math
import cv2

from nadirshape.misc.epc import EPC
from nadirshape.misc.atlanta_transform import E2P

PI = float(np.pi)

def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * PI


def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * PI

#####MINE transform
def np_coor2xy(coor, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512, m_ratio = 1.0):
    '''
    coor: N x 2, index of array in (col, row) format eg. 1024x2
    m_ratio: pixel/cm ratio for tensor fitting
    '''           
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)       
    v = np_coory2v(coor[:, 1], coorH)
                  
    c = z / np.tan(v)
    x = m_ratio * c * np.sin(u) + floorW / 2.0 - 0.5
    y = -m_ratio *c * np.cos(u) + floorH / 2.0 - 0.5
    
    return np.hstack([x[:, None], y[:, None]])

def xy2coor(xy, z=50, coorW=1024, coorH=512, eps = 0.001):
    '''
    xy: N x 2
    '''
    x = xy[:, 0] + eps ##- floorW / 2 + 0.5
    y = xy[:, 1] + eps##- floorH / 2 + 0.5

    u = np.arctan2(x, -y)
    v = np.arctan(z / np.sqrt(x**2 + y**2))

    coorx = (u / (2 * PI) + 0.5) * coorW - 0.5
    coory = (-v / PI + 0.5) * coorH - 0.5

    return np.hstack([coorx[:, None], coory[:, None]])


class D2L(nn.Module):
    def __init__(self, gpu=False, H = 512, W = 1024, fp_size = 512, fp_fov = 165.0):
        super(D2L, self).__init__()

        self.fp_size = fp_size
        self.fp_fov = fp_fov

        self.img_size = [W,H]

        self.ep = EPC(gpu=gpu)
        self.e2p = E2P(equ_size=(H, W), out_dim=self.fp_size, fov=self.fp_fov, radius=1, gpu = gpu, return_fl = True)

        self.xz_sph = self.ep.atlanta_sphere(H, W)

    def get_segmentation_masks(self, seg_pred):
        soft_sem = torch.softmax(seg_pred, dim = 1) #####TO DO - here semantic is given by clutter mask
        soft_sem = torch.argmax(soft_sem, dim=1, keepdim=True)
        soft_sem = torch.clamp(soft_sem, min=0, max=1)
        masks = torch.zeros_like(seg_pred).to(seg_pred.device)
        masks.scatter_(1, soft_sem, 1)
                        
        return masks

    def get_translated_layout_edges(self, src_depth, src_layout_seg, x_c):#######batched        
        mask_pred = self.get_segmentation_masks(src_layout_seg)
        layout_mask = mask_pred[:,:1]##

        B,H,W = src_depth.size()
        
        LE = []

        for i in range(B):
            ##print('element batch shape', src_depth[i:i+1].shape, self.xz_sph.shape)
            p_max, p_min = self.max_min_depth(src_depth[i:i+1])

            p_max = p_max.cpu().numpy()
            p_min = p_min.cpu().numpy()
                        
            m_contour = self.contour_from_cmask(layout_mask.cpu(), epsilon_b=0.01)####return numpy contour
                                                
            ##equi_pts, equi_1D, cart_XY = dataset.d2l.contour_pts2equi_layout(m_contour, x_img.shape[3], x_img.shape[2], p_max, p_min, theta_sort = False)
            equi_pts, equi_1D, cart_XY = self.contour_pts2equi_layout(m_contour, W, H, p_max, p_min, translation = x_c.squeeze(0).numpy())

            le = get_layout_edges(torch.FloatTensor(equi_pts).unsqueeze(0), H, W)

            le = torch.FloatTensor(le).unsqueeze(0).unsqueeze(0)

            LE.append(le)

        layout_edges = torch.cat(LE, dim=0)####Bx1xhxw
                        
    
        return layout_edges

    def get_layout(self, src_depth, src_layout_seg):#######batched        
        mask_pred = self.get_segmentation_masks(src_layout_seg)
        layout_mask = mask_pred[:,:1]##

        B,H,W = src_depth.size()
        
        LE = []

        for i in range(B):
            ##print('element batch shape', src_depth[i:i+1].shape, self.xz_sph.shape)
            p_max, p_min = self.max_min_depth(src_depth[i:i+1])

            p_max = p_max.cpu().numpy()
            p_min = p_min.cpu().numpy()
                        
            m_contour = self.contour_from_cmask(layout_mask.cpu(), epsilon_b=0.01)####return numpy contour
                                                
            ##equi_pts, equi_1D, cart_XY = dataset.d2l.contour_pts2equi_layout(m_contour, x_img.shape[3], x_img.shape[2], p_max, p_min, theta_sort = False)
            equi_pts, equi_1D, cart_XY = self.contour_pts2equi_layout(m_contour, W, H, p_max, p_min)

            le = torch.FloatTensor(equi_pts).unsqueeze(0)
                        
            LE.append(le)

        layouts = torch.cat(LE, dim=0)####Bx1xhxwtorch.cos
                        
    
        return layouts
    
    ################from c-f bondaries returns (eventually translated): layout atlanta mask, xy coords, max, min, e_pts  
    def atlanta_transform_from_equi_corners(self, e_pts, max, min, translation = np.zeros((1,3)), theta_sort = False, ceiling_mask = True): 
                                                           
        ###spitting ceiling and floor
        W = self.img_size[0]
        H = self.img_size[1]  
                       
        c_pts_x = []
        c_pts_y = []

        n_cor = len(e_pts)
        for i in range(n_cor // 2):
            c_pts_x.append(e_pts[i*2][0])
            c_pts_y.append(e_pts[i*2][1])
                        
        c_pts_x = np.array(c_pts_x)
        c_pts_y = np.array(c_pts_y)

        c_pts = np.stack((c_pts_x, c_pts_y), axis=-1)#########ceiling equi corners (N,2)
        #################################################

        ####convert to cartesian XY
        XY = np_coor2xy(c_pts, max, W, H, floorW=1.0, floorH=1.0)
                
        #####translate
        x_tr = translation[:,:1]
        y_tr = translation[:,1:2]
        z_tr  = translation[:,2:]               
        XY[:,:1]  -= x_tr
        XY[:,1:2] += y_tr
        max -= z_tr.squeeze(0)
        min += z_tr.squeeze(0)
                
                
        ####create translated equi coords
        ####default: cartesian order
        tr_c_cor = xy2coor(np.array(XY),  max, W, H)
        tr_f_cor = xy2coor(np.array(XY), -min, W, H) ####based on the ceiling shape
                
        cor_count = len(tr_c_cor)                                                      
                               
        if(theta_sort):
            ####sorted by theta (pixels coords)
            c_ind = np.lexsort((tr_c_cor[:,1],c_cor[:,0])) 
            f_ind = np.lexsort((tr_f_cor[:,1],f_cor[:,0]))
            tr_c_cor = tr_c_cor[c_ind]
            tr_f_cor = tr_f_cor[f_ind]
                       
        tr_equi_coords = []
                
        for j in range(len(tr_c_cor)):
            tr_equi_coords.append(tr_c_cor[j])
            tr_equi_coords.append(tr_f_cor[j])
               
        tr_e_pts = np.array(tr_equi_coords)
                                
        if(ceiling_mask):
            z_dist = max
        else:
            z_dist = min

        metric2fp = self.fp_size / (z_dist / math.tan(math.pi *  (180 - self.fp_fov) / 360) * 2)
        
        atl_xy = np_coor2xy(tr_c_cor, max, W, H, floorW=self.fp_size, floorH=self.fp_size, m_ratio = metric2fp)###NB. converintg to footprint (wrong func name)
              

        ###create atlanta mask
        mask = np.zeros([self.fp_size, self.fp_size], np.uint8) 
        ##mask = np.zeros([self.fp_size, self.fp_size], np.single)                   
        m_pts = atl_xy.astype(np.int32)
        m_pts = m_pts.reshape((-1,1,2))
        cv2.fillPoly(mask, [m_pts], 1)                                 
                   

        return torch.FloatTensor(mask), torch.FloatTensor(XY), max, min, torch.FloatTensor(tr_e_pts)
            
    def max_min_depth(self, src_depth):
        src_depth_plan = self.ep.euclidean_to_planar_depth(src_depth.squeeze(0), self.xz_sph).unsqueeze(0)##

        ##print(src_depth_plan.shape)

        B,_,H,W = src_depth_plan.size()

        up_depth, bottom_depth = torch.split(src_depth_plan, H//2, dim=2)

        max = torch.max(up_depth)
        min = torch.max(bottom_depth)
               
        ##print(max, min)

        return max,min

    def convert_depth_mapping(self, src_depth, sphere_type = 'polar'):
        if(sphere_type == 'atlanta'):
            xz_sph = self.ep.atlanta_sphere(self.img_size[1], self.img_size[0])

        if(sphere_type == 'polar'):
            xz_sph = self.ep.polar_sphere(self.img_size[1], self.img_size[0])

        if(sphere_type == 'euclidean'):
            xz_sph = self.ep.xyz_sphere(self.img_size[1], self.img_size[0])
        
        src_depth_plan = self.ep.euclidean_to_planar_depth(src_depth.squeeze(0), xz_sph).unsqueeze(0)##

        return src_depth_plan
        
    def atlanta_transform_from_depth(self, src_depth):
        ####input depth 1xhxw - no batched
        src_depth_plan = self.ep.euclidean_to_planar_depth(src_depth.squeeze(0), self.xz_sph).unsqueeze(0)## input: ###### (hxw) - (1xhxw)
                                        
        [d_up, d_down, fl] = self.e2p(src_depth_plan) ####batched trans - same fl

        c_dist = torch.max(d_up)
        f_dist = torch.max(d_down)

       
           
        return d_up, d_down, c_dist, f_dist, fl

    def batched_atlanta_transform_from_depth(self, src_depth):
        ####input depth Bx1hxw - batched
        batch_size = src_depth.size()[0]
        
        DP = []

        for i in range(batch_size):
            ##print('element batch shape', src_depth[i:i+1].shape, self.xz_sph.shape)
            dp = self.ep.euclidean_to_planar_depth(src_depth[i:i+1], self.xz_sph) ## input: ###### (1xhxw) - (1xhxw)
            DP.append(dp)

        src_depth_plan = torch.cat(DP, dim=0)####Bx1xhxw
                
        ##src_depth_plan = self.ep.euclidean_to_planar_depth(src_depth.squeeze(0), self.xz_sph).unsqueeze(0)## input: 1xhxw , 1xwxh
                        
        [d_up, d_down, fl] = self.e2p(src_depth_plan) ####batched trans - same fl

        c_dist = torch.max(d_up)
        f_dist = torch.max(d_down)

       
           
        return d_up.to(src_depth.device), d_down.to(src_depth.device), c_dist.to(src_depth.device), f_dist.to(src_depth.device)
    
        
    def cmask_from_depth(self, src_depth):
        
        d_up, d_down, c_dist, f_dist, fl = self.atlanta_transform_from_depth(src_depth)                          
        
        c_th = c_dist * 0.95

        cmask = (d_up > c_th).float()                

       
        return cmask, c_dist, f_dist, fl

    def atl_pts2xy(self, fp_pts, f_dist, as_numpy = True, canonical_axis = False, fp_size = None):
        
        if(fp_size is None):
            fp_size = self.fp_size

        if(as_numpy):
            metric_scale = (f_dist / math.tan(math.pi *  (180 - self.fp_fov) / 360) * 2).detach().numpy()
        else:
            metric_scale = (f_dist / math.tan(math.pi *  (180 - self.fp_fov) / 360) * 2).to(fp_pts.device)

        ###from pixels to norm cartesian coords
        if(as_numpy):
            fp_pts = fp_pts.astype(float)
        else:        
            fp_pts = fp_pts.type(torch.FloatTensor).to(fp_pts.device) 

        fp_pts -= fp_size / 2
        fp_pts /= fp_size
        

        if(canonical_axis and (not as_numpy)):
            fp_u, fp_v = torch.unbind(fp_pts, dim=2)

            fp_u *= -1
                      
            fp_pts = torch.cat([fp_v,fp_u],dim=1).unsqueeze(1)
                        
                           
        for i in range(fp_pts.shape[1]):
            fp_xy = fp_pts[:,i] * metric_scale

            ##print('testing point', fp_xy)
                        
        return fp_xy

    
    def xy2atl(self, xy, global_fp, z_dist):
        metric2fp = 1.0 / (z_dist / math.tan(math.pi *  (180 - self.fp_fov) / 360) * 2).detach().numpy()

        ###FIXME
        ##xy[1] *= -1
        xy = np.flip(xy)

        fp_trans = xy * metric2fp * (float(self.fp_size)/float(global_fp))

        fp_trans *= global_fp#####TO BE REMOVED - see before

        fp_trans += global_fp / 2



        return fp_trans

    def nadir2xy(self, fp_pts, global_fp, z_dist):
        ###fp_pts.shape (N,1,2)
                
        fp2metric = (z_dist / math.tan(math.pi *  (180 - self.fp_fov) / 360) * 2).detach().cpu().numpy()       
        
        fp_pts = fp_pts.astype(float)

        fp_pts[:,:,0] = global_fp - fp_pts[:,:,0]

        
                        
        fp_pts -= global_fp / 2.0
                                
        xy = fp_pts  * fp2metric / float(self.fp_size)

        xy = np.flip(xy)

        
                                       
        return xy

    
    def insert_local_transform(self, local_fp, world_fp, z_dist, xy_trans):
        ###input local: Bx1xLfpxLfp output: world Bx1xWfpxWfp

        metric2fp = 1.0 / (z_dist / math.tan(math.pi *  (180 - self.fp_fov) / 360) * 2)

        B, _, loc_fp_h, loc_fp_w = local_fp.size()

        _, _, world_fp_h, world_fp_w = world_fp.size()

                        
        fp_trans = xy_trans * metric2fp ##* (float(loc_fp_w))##/float(world_fp_w))

        fp_trans *= loc_fp_w

        fp_trans += world_fp_w / 2

                              
        u = abs(int((world_fp_w-fp_trans[:,1])-(loc_fp_w/2)))###to local center
        v = abs(int(fp_trans[:,0]-(loc_fp_h/2)))

        ##print('debug', u, loc_fp_h+u, v, loc_fp_w+v)
               
        world_fp[:,:, u:loc_fp_h+u, v:loc_fp_w+v] = local_fp


    
    def global_transform(self, x_img, x_depth, net, floor_W = 1024, floor_H = 1024):
        ##
        d_up, d_down, c_dist, f_dist, fl = self.atlanta_transform_from_depth(torch.FloatTensor(x_depth).unsqueeze(0))#####input 1xhxw
        ##print('output atl depth', x_atl_depth.shape, x_atl_depth.device)
        ##layout = self.layout_from_depth(x_atl_depth) ### input: Bx1xhxh
                      
        plt.figure(1030)
        plt.title(' up depth src')
        plt.imshow(d_up.squeeze(0).squeeze(0)) 

        plt.figure(1031)
        plt.title(' down depth src')
        plt.imshow(d_down.squeeze(0).squeeze(0)) 


        depth, mask_pred = inference(net, x_img, device)

        mask_pred = self.get_segmentation_masks(d_up)##FIXMEEEEE need prediction to extract the mask!!!!

        layout_mask = mask_pred[:,:1]
   
        p_max, p_min = self.max_min_depth(x_depth)

        p_max = p_max.cpu().numpy()
        p_min = p_min.cpu().numpy()
                        
        m_contour = self.contour_from_cmask(layout_mask.cpu(), epsilon_b=0.01)####return numpy contour                                     
   
        cart_XY, Z_up, Z_down  = self.contour_pts2xyz_layout(m_contour, p_max, p_min)

        pre_poly = Polygon(cart_XY)####NB polygon convert xy coords to image coords (1024x1024)
                           
                                        
        plt.figure(141)
        plt.title('XY pre')
        plt.gca().invert_yaxis()    
        plt.axes().set_aspect('equal')
        plt.plot(*pre_poly.exterior.xy,color='red',alpha=0.8)

    ###########valid method from here
    def contour_pts2equi_layout( self, c_pts, W, H, c_dist, f_dist, theta_sort = False, translation = np.zeros((1,3)) ):####NB. use theta sort to store 2D footprint
        ##c_pts = c_pts.squeeze(1) 

        xy = self.atl_pts2xy(c_pts, c_dist)
                        
        ####apply translation
        ##print(xy.shape, translation.shape)
        x_tr = translation[:,:1]
        y_tr = translation[:,1:2]
        z_tr  = translation[:,2:]

        ##print(xy_tr, z_tr)
        ##print('tr input',xy)

        xy[:,:1]  -= x_tr
        xy[:,1:2] += y_tr

        off = (z_tr.squeeze(0))[0]
                
        c_dist -= off              
        f_dist += off
        ##print('tr output',xy)
        
        ####default: cartesian order
        c_cor = xy2coor(np.array(xy),  c_dist, W, H)
        f_cor = xy2coor(np.array(xy), -f_dist, W, H) ####based on the ceiling shape

        cor_count = len(c_cor)                                                      
                               
        if(theta_sort):
            ####sorted by theta (pixels coords)
            c_ind = np.lexsort((c_cor[:,1],c_cor[:,0])) 
            f_ind = np.lexsort((f_cor[:,1],f_cor[:,0]))
            c_cor = c_cor[c_ind]
            f_cor = f_cor[f_ind]
                       
        equi_coords = []
                
        for j in range(len(c_cor)):
            equi_coords.append(c_cor[j])
            equi_coords.append(f_cor[j])
               
        equi_coords = np.array(equi_coords)
        xs = np.array(c_cor) 

        u,v = torch.unbind(torch.from_numpy(xs), dim=1)
                    
        return equi_coords, u.numpy().astype(np.uint32), xy

    def contour_pts2xyz_layout( self, c_pts, c_dist, f_dist, use_floor = False, translation = np.zeros((1,3)) ):####NB. use theta sort to store 2D footprint
        #
        dist = c_dist

        if (use_floor):
            dist = f_dist

        xy = self.atl_pts2xy(c_pts, dist)
                        
        ####apply translation
        ##print(xy.shape, translation.shape)

        x_tr = translation[:,:1]
        y_tr = translation[:,1:2]
        z_tr  = translation[:,2:]

        ##print(xy_tr, z_tr)
        ##print('tr input',xy)

        xy[:,:1]  -= x_tr
        xy[:,1:2] += y_tr

        off = (z_tr.squeeze(0))[0]
                
        c_dist -= off              
        f_dist += off              
                                    
        return xy, c_dist, f_dist       

    def contour_from_cmask(self, mask, epsilon_b=0.005, get_valid = False):
        
        data_cnt, data_heri = cv2.findContours(np.uint8(mask.squeeze(0).squeeze(0)), 1, 2)##CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE
        ##data_cnt, data_heri = cv2.findContours(data_thresh, 0, 2)##CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE
        
        is_valid = False
        
        approx = np.empty([1, 1, 2])

        #data_cnt.sort(key=lambda x: cv2.contourArea(x), reverse=True)

        #if(len(data_cnt)>0):
        #    epsilon = epsilon_b*cv2.arcLength(data_cnt[0], True)
        #    approx = cv2.approxPolyDP(data_cnt[0], epsilon, True)
        #    is_valid = True
        #else:
        #    print('WARNING: no contour found')   

        if(len(data_cnt)>0):
            c = max(data_cnt, key = cv2.contourArea)
            epsilon = epsilon_b*cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            is_valid = True
        else:
            print('WARNING: no contour found')   

        ######DEBUG
        #draw_mask = np.uint8(mask.squeeze(0).squeeze(0))                        
        #cv2.polylines(approx, [approx], True, (255),2,cv2.LINE_AA)

        #plt.figure(1034)
        #plt.title('contour')
        #plt.imshow(draw_mask)
        #################################################
            
        if(get_valid):                
            return approx, is_valid
        else:
            return approx
             

    def forward(self, depth, tr = None):                   
        ###
        mask, max, min, fl = self.cmask_from_depth(depth)
        print('max', max, 'min', min, 'height', max+min, 'focal l', fl)

        m_contour = self.contour_from_cmask(mask)

        #draw_mask = np.uint8(mask.squeeze(0).squeeze(0))                        
        #cv2.polylines(draw_mask, [m_contour], True, (255),2,cv2.LINE_AA)

        #plt.figure(1034)
        #plt.title('contour')
        #plt.imshow(draw_mask)

        tr_in = np.zeros((1,3))

        if(tr is not None):
            tr_in = tr.numpy()

        use_post_proc = True

        if(use_post_proc):
            equi_pts, equi_1D, cart_XY = self.contour_pts2equi_layout(m_contour, self.img_size[0], self.img_size[1], max.numpy(), min.numpy(), theta_sort = False, translation = tr_in)
                                
            #pre_poly = Polygon(cart_XY)####NB polygon convert xy coords to image coords (1024x1024)
                           
                                        
            #plt.figure(141)
            #plt.title('XY pre')
            #plt.gca().invert_yaxis()    
            #plt.axes().set_aspect('equal')
            #plt.plot(*pre_poly.exterior.xy,color='red',alpha=0.8)
                        
                               
            ###using post_processing

            print(equi_pts.shape)

            y_bon_ = cor_2_1d(equi_pts, self.img_size[1], self.img_size[0])

            equi_pts, xy_cor = MW_post_processing(equi_1D, y_bon_, self.img_size[0], self.img_size[1], max.numpy(), -min.numpy(), post_force_cuboid = False)

            #post_poly = Polygon(xy_cor)
            #plt.figure(142)
            #plt.title('XY post')
            #plt.gca().invert_yaxis()    
            #plt.axes().set_aspect('equal')
            #plt.plot(*post_poly.exterior.xy,color='green')
                       
        else:               
            equi_pts, equi_1D, cart_XY = self.contour_pts2equi_layout(m_contour, self.img_size[0], self.img_size[1], max.numpy(), min.numpy(), theta_sort = False, translation = tr_in)  
                                                        
        ##print('post proc input',xs,y_bon_)            
               
        ##DL = torch.cat(DL, dim=0) ####to torch batch

        return torch.FloatTensor(equi_pts),max,min 