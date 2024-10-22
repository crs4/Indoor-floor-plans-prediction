import argparse
import os
import json
from tqdm import tqdm
import numpy as np
import torch.utils.data
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import cv2

##from nadir_floorplan_creator import NadirTransformsCreator

###FIXME
##from data_preprocess.common_utils import export_density
##from data_preprocess.stru3d.stru3d_utils import normalize_annotations, parse_floor_plan_polys, generate_coco_dict
##from datasets import poly_data

###NEW
from nadirshape.nadirshapenet import NadirShapeNet, load_trained_model
from nadirshape.misc.d2l import D2L
import matplotlib.pyplot as plt


######adapted from RoomFormer##############################################################
###########################################################################################
invalid_scenes_ids = [76, 183, 335, 491, 663, 681, 703, 728, 865, 936, 985, 986, 1009, 1104, 1155, 1221, 1282, 
                     1365, 1378, 1635, 1745, 1772, 1774, 1816, 1866, 2037, 2076, 2274, 2334, 2357, 2580, 2665, 
                     2706, 2713, 2771, 2868, 3156, 3192, 3198, 3261, 3271, 3276, 3296, 3342, 3387, 3398, 3466, 3496]

type2id = {'living room': 0, 'kitchen': 1, 'bedroom': 2, 'bathroom': 3, 'balcony': 4, 'corridor': 5,
            'dining room': 6, 'study': 7, 'studio': 8, 'store room': 9, 'garden': 10, 'laundry room': 11,
            'office': 12, 'basement': 13, 'garage': 14, 'undefined': 15, 'door': 16, 'window': 17}

def normalize_point(point, normalization_dict):

    min_coords = normalization_dict["min_coords"]
    max_coords = normalization_dict["max_coords"]
    image_res = normalization_dict["image_res"]

    ##print('points',min_coords, max_coords)

    point_2d = \
        np.round(
            (point[:2] - min_coords[:2]) / (max_coords[:2] - min_coords[:2]) * image_res)
    point_2d = np.minimum(np.maximum(point_2d, np.zeros_like(image_res)),
                            image_res - 1)

    point[:2] = point_2d.tolist()

    return point

def normalize_annotations(scene_path, normalization_dict, json_name="annotation_3dm.json"):
    annotation_path = os.path.join(scene_path, json_name)

    ##print('annotation_path',annotation_path)

    with open(annotation_path, "r") as f:
        annotation_json = json.load(f)

    for line in annotation_json["lines"]:
        point = line["point"]
        point = normalize_point(point, normalization_dict)
        line["point"] = point

    for junction in annotation_json["junctions"]:
        point = junction["coordinate"]
        point = normalize_point(point, normalization_dict)
        junction["coordinate"] = point

    return annotation_json

def parse_floor_plan_polys(annos):
    planes = []
    for semantic in annos['semantics']:
        for planeID in semantic['planeID']:
            if annos['planes'][planeID]['type'] == 'floor':
                ##print('parse_floor_plan_polys',semantic['type'],annos['planes'][planeID]['type'])
                planes.append({'planeID': planeID, 'type': semantic['type']})

        if semantic['type'] == 'outwall':
            outerwall_planes = semantic['planeID']

    # extract hole vertices
    lines_holes = []
    for semantic in annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            #
            for planeID in semantic['planeID']:
                lines_holes.extend(np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist())
    lines_holes = np.unique(lines_holes)

    # junctions on the floor
    junctions = np.array([junc['coordinate'] for junc in annos['junctions']])
    junction_floor = np.where(np.isclose(junctions[:, -1], 0))[0]

    # construct each polygon
    polygons = []

    for plane in planes:
        ##print('debug str3d_utils',plane['type'])####NB. label is undefined.....
        lineIDs = np.where(np.array(annos['planeLineMatrix'][plane['planeID']]))[0].tolist()
        junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
        polygon = convert_lines_to_vertices(junction_pairs)
        polygons.append([polygon[0], plane['type']])

            
    return polygons

def convert_lines_to_vertices(lines):
    """
    convert line representation to polygon vertices

    """
    polygons = []
    lines = np.array(lines)
        
    polygon = None

    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)

        lineID, juncID = np.where(lines == polygon[-1])

        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID, 0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    return polygons

def generate_coco_dict(annos, polygons, curr_instance_id, curr_img_id, ignore_types, filter_openings=True):

    junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])

    coco_annotation_dict_list = []

    for poly_ind, (polygon, poly_type) in enumerate(polygons):
        if poly_type in ignore_types:
            continue

        polygon = junctions[np.array(polygon)]

        poly_shapely = Polygon(polygon)
        area = poly_shapely.area

        ##print('poly_shapely area', area)

        # assert area > 10
        # if area < 100:
        if poly_type not in ['door', 'window'] and area < 100:
            continue

        if poly_type in ['door', 'window'] and area < 10:
            continue
        
        rectangle_shapely = poly_shapely.envelope

        ### here we convert door/window annotation into a single line
        if poly_type in ['door', 'window']:
            assert polygon.shape[0] == 4
            midp_1 = (polygon[0] + polygon[1])/2
            midp_2 = (polygon[1] + polygon[2])/2
            midp_3 = (polygon[2] + polygon[3])/2
            midp_4 = (polygon[3] + polygon[0])/2

            dist_1_3 = np.square(midp_1 -midp_3).sum()
            dist_2_4 = np.square(midp_2 -midp_4).sum()
            if dist_1_3 > dist_2_4:
                polygon = np.row_stack([midp_1, midp_3])
            else:
                polygon = np.row_stack([midp_2, midp_4])

        coco_seg_poly = []
        poly_sorted = resort_corners(polygon)

        for p in poly_sorted:
            coco_seg_poly += list(p)

        # Slightly wider bounding box
        bound_pad = 2
        bb_x, bb_y = rectangle_shapely.exterior.xy
        bb_x = np.unique(bb_x)
        bb_y = np.unique(bb_y)
        bb_x_min = np.maximum(np.min(bb_x) - bound_pad, 0)
        bb_y_min = np.maximum(np.min(bb_y) - bound_pad, 0)

        bb_x_max = np.minimum(np.max(bb_x) + bound_pad, 256 - 1)
        bb_y_max = np.minimum(np.max(bb_y) + bound_pad, 256 - 1)

        bb_width = (bb_x_max - bb_x_min)
        bb_height = (bb_y_max - bb_y_min)

        ##print('coco bbox',bb_width,bb_height)

        coco_bb = [bb_x_min, bb_y_min, bb_width, bb_height]

        polygon_ratio = bb_width / bb_height

        ##print('polygon ratio', bb_width / bb_height)
                
        coco_annotation_dict = {
                "segmentation": [coco_seg_poly],
                "area": area,
                "iscrowd": 0,
                "image_id": curr_img_id,
                "bbox": coco_bb,
                "category_id": type2id[poly_type],
                "id": curr_instance_id}

        if( (polygon_ratio < 60 or polygon_ratio > 0.01)  or not filter_openings):        
            coco_annotation_dict_list.append(coco_annotation_dict)
            curr_instance_id += 1


    return coco_annotation_dict_list

def is_clockwise(points):
    # points is a list of 2d points.
    assert len(points) > 0
    s = 0.0
    for p1, p2 in zip(points, points[1:] + [points[0]]):
        s += (p2[0] - p1[0]) * (p2[1] + p1[1])
    return s > 0.0

def resort_corners(corners):
    # re-find the starting point and sort corners clockwisely
    x_y_square_sum = corners[:,0]**2 + corners[:,1]**2 
    start_corner_idx = np.argmin(x_y_square_sum)

    corners_sorted = np.concatenate([corners[start_corner_idx:], corners[:start_corner_idx]])

    ## sort points clockwise
    if not is_clockwise(corners_sorted[:,:2].tolist()):
        corners_sorted[1:] = np.flip(corners_sorted[1:], 0)

    return corners

def export_density(density_map, out_folder, scene_id):
    density_path = os.path.join(out_folder, scene_id+'.png')
    density_uint8 = (density_map * 255).astype(np.uint8)
    cv2.imwrite(density_path, density_uint8)


##########################################################################################
##########################################################################################

class NadirTransformsCreator():
###
    def __init__(self, path, net, d2l, device = 'cpu', encode_heightmap = False, data_type = 's3d', room_type = "full"):
        self.path = path
        
        self.net = net
        self.d2l = d2l

        self.data_type = data_type        

         #####input image is a full-cluttered image        

        print('NadirTransformsCreator data type', self.data_type)        

        if(self.data_type == 'zind'):
            ####NB. only full is available
            
            self.camera_h = 1700.0 ##default##mm - ONLY to restore metric scale for zind          
            self.metric_scale = self.camera_h #####default

            self.scale_zind_annotations(scene_path = self.path, metric_scale = self.metric_scale) #####override with metrically scaled annotations
        
        sections = [p for p in os.listdir(os.path.join(path, "2D_rendering"))]
        self.rgb_paths = [os.path.join(*[path, "2D_rendering", p, "panorama", room_type, "rgb_rawlight.png"]) for p in sections]
        ##self.rgb_paths = [os.path.join(*[path, "2D_rendering", p, "panorama", "full", "rgb_coldlight.png"]) for p in sections]
        self.camera_paths = [os.path.join(*[path, "2D_rendering", p, "panorama", "camera_xyz.txt"]) for p in sections]
        self.max_min_path = os.path.join(path, "metric_max_min.json")

                       
        self.device = device
                           
        
        self.point_cloud = None

        self.encode_heightmap = encode_heightmap               
        
        self.camera_centers = self.read_camera_center()#####NB. doing it first

    
    def scale_zind_annotations(self, scene_path, metric_scale, file_in="annotation_3d.json", file_out="annotation_3dm.json"):
        annotation_path = os.path.join(scene_path, file_in)
        annotation_path_metric = os.path.join(scene_path, file_out)

        ##print('annotation_path',annotation_path)

        with open(annotation_path, "r") as f:
            annotation_json = json.load(f)

        for line in annotation_json["lines"]:
            point = line["point"]                        
            point = metric_scale * np.asarray(point).astype(float)##normalize_point(point, normalization_dict)
            line["point"] = point.tolist()

        for junction in annotation_json["junctions"]:
            point = junction["coordinate"]
            point = metric_scale * np.asarray(point).astype(float)##normalize_point(point, normalization_dict)
            junction["coordinate"] = point.tolist()

        with open(annotation_path_metric, 'w') as f:
                json.dump(annotation_json, f)    

    def read_camera_center(self):
        camera_centers = []
        for i in range(len(self.camera_paths)):
            if(self.data_type == 's3d'):
                with open(self.camera_paths[i], 'r') as f:
                    line = f.readline()
                center = list(map(float, line.strip().split(" ")))
                camera_centers.append(np.asarray([center[0], center[1], center[2]]))
            else:
                ##print('DEBUG read_camera_center', self.camera_paths[i])                
                with open(self.camera_paths[i], 'r') as f:
                    line1 = f.readline()
                                        
                    c0 = float(line1)#*self.camera_h
                    line2 = f.readline()
                    c1 = float(line2)#*self.camera_h
                    line3 = f.readline()
                    c2 = float(line3)#*self.camera_h

                ##center = list(map(float, line.strip().split(" ")))
                camera_centers.append(np.asarray([c0, c1, c2]))
        return camera_centers  

    def generate_floorplan_map(self, width=2048, height=2048, out_w = 256, out_h = 256, force_camera_h = True, get_normalization = False):####NB. scale to 256 for comaptibility with roomformer
        image_res = np.array((out_w, out_h))##########NB. output pixel dims
           
        local_center = torch.zeros(1,3).to(self.device)

        nadir_map = []
                                                        
        # Getting rooms
        for i in range(len(self.rgb_paths)):
                
            rgb_img = Image.open(self.rgb_paths[i])

            if(self.data_type == 'zind'):
                rgb_img = rgb_img.resize((1024,512), Image.BICUBIC)

            img = np.array(rgb_img, np.float32)[..., :3] / 255.     
            
            x_img = torch.FloatTensor(img.transpose([2, 0, 1])).unsqueeze(0)###as batch element

            depth, transform = self.net(x_img.to(self.device))

            mask_pred = self.get_segmentation_masks(transform)

            nadir_mask_pred = mask_pred[:,:1]##

            # plt.figure(i)
            # plt.title('pred mask'+str(i))
            # plt.imshow(nadir_mask_pred.cpu().squeeze(1).squeeze(0)) 
                        

            ####NEW
            if(self.encode_heightmap):
                heightmap = self.room_heightmap(depth.unsqueeze(1), nadir_mask_pred)

            #print('room heightmap', heightmap.shape)

            #plt.figure(i+3)
            #plt.title('ceiling heightmap'+str(i))
            #plt.imshow(heightmap) 

           
            ###read camera center
            if(self.data_type == 'zind'):
                rc = (self.camera_centers[i]*self.camera_h) / 1000.0 ###to meters
            else:
                rc = self.camera_centers[i] / 1000.0 ###to meters

            rc = torch.from_numpy(rc).unsqueeze(0).to(self.device)

            ### move to local center
            if(i == 0):
                local_center = rc
                                               
            
            xy_trans = rc - local_center
                
            p_max, p_min = self.d2l.max_min_depth(depth)
                                   
            nm = torch.zeros(1, 1, height, width)
                
            z_dist = p_min

            if(force_camera_h):
                z_dist = rc[:,2]

                
            ###encoding room-wise
            nadir_mask_pred *= (10*(i+1))#####cam code
                                
              
            if(self.encode_heightmap):
                self.d2l.insert_local_transform(heightmap.unsqueeze(0).unsqueeze(0), nm, z_dist, xy_trans)
            else:
                self.d2l.insert_local_transform(nadir_mask_pred, nm, z_dist, xy_trans)
           
            nadir_map.append(nm)

                        
        nadir_map = torch.cat(nadir_map, dim=1)       
        nadir_map,_ = torch.max(nadir_map,dim=1)####flattened to the same map
                                            
        #####convert annotations to montefloor-roomformer format
        uv = (torch.nonzero(nadir_map.squeeze(0).squeeze(0))).numpy()#### (N,2)
        
        max_coords = np.max(uv, axis=0)
        min_coords = np.min(uv, axis=0)
        max_m_min = max_coords - min_coords
        
        ####NB. store bbox
        max_coords2 = max_coords[np.newaxis,np.newaxis, :]
        min_coords2 = min_coords[np.newaxis,np.newaxis, :]

        uv_max_min = np.concatenate((max_coords2, min_coords2), axis=0)
        ###############################################

        max_coords = (max_coords + 0.1 * max_m_min).astype(np.int32)###adding border
        min_coords = (min_coords - 0.1 * max_m_min).astype(np.int32)###adding border                  
        
        ###crop and scale density map
        np_crop = nadir_map.cpu().squeeze(1).squeeze(0).numpy()

        w_crop = max_coords[0]-min_coords[0]
        h_crop = max_coords[1]-min_coords[1]
            
        np_crop = np_crop[min_coords[0]:min_coords[0]+w_crop,min_coords[1]:min_coords[1]+h_crop]
                      
        np_crop = np.asarray(self.scale(np.flipud(np_crop), out_w, out_h), np.float32)#####convert to montefloor-roomformer format
                            
        # plt.figure(111)
        # plt.title('transform map')
        # plt.imshow(nadir_map.cpu().squeeze(1).squeeze(0)) 

        # plt.figure(112)
        # plt.title('floor map')
        # plt.imshow(np_crop) 

        # plt.show() 
        
        result = []

        result.append(np_crop)

        if(get_normalization):
            normalization_dict = {}
            ####convert nadir coords to metric
                        
            ##print('DEBUG saving metric infor: uv_max_min',uv_max_min, 'ceil floor dist',p_max, p_min, 'footprint w', width)

            max_min = self.d2l.nadir2xy(uv_max_min, width, z_dist)

            max_min = max_min.squeeze(1)####(2,2)

            ###NB. flip max with min
            metric_min_coords = np.array([max_min[0,0],max_min[1,1],0.0])+local_center.squeeze(0).cpu().numpy()
            metric_max_coords = np.array([max_min[1,0],max_min[0,1],0.0])+local_center.squeeze(0).cpu().numpy()  
            #                   
                        
            max_m_min = metric_max_coords - metric_min_coords
            
            metric_max_coords = metric_max_coords + 0.1 * max_m_min###adding border
            metric_min_coords = metric_min_coords - 0.1 * max_m_min###adding border

            normalization_dict["min_coords"] = metric_min_coords * 1000.0 ###to mm
            normalization_dict["max_coords"] = metric_max_coords * 1000.0 ###to mm
            normalization_dict["image_res"] = image_res
            normalization_dict["map_res"] = [width,height]
            normalization_dict["heights"] = [p_max.detach().cpu().numpy().item(), p_min.detach().cpu().numpy().item()]
            
            result.append(normalization_dict)
           
        return result

    def get_segmentation_masks(self, seg_pred):
            soft_sem = torch.softmax(seg_pred, dim = 1) #####TO DO - here semantic is given by clutter mask
            soft_sem = torch.argmax(soft_sem, dim=1, keepdim=True)
            soft_sem = torch.clamp(soft_sem, min=0, max=1)
            masks = torch.zeros_like(seg_pred).to(seg_pred.device)
            masks.scatter_(1, soft_sem, 1)
                    
            return masks 

    def scale(self, im, nR, nC):
          nR0 = len(im)     # source number of rows 
          nC0 = len(im[0])  # source number of columns 

          print('scaling',nR0,nC0)

          return [[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]  
                     for c in range(nC)] for r in range(nR)]

    def room_heightmap(self,b_depth, layout_mask):
        ####NEW PRED DEPTH transforms
        d_ceiling, d_floor, c_dist, f_dist = self.d2l.batched_atlanta_transform_from_depth(b_depth)#####input Bx1xhxw

        fc_ration = c_dist / f_dist
                
        ##d_ceiling = self.resize_crop(d_ceiling.squeeze(0).squeeze(0).numpy(),fc_ration, layout_mask.shape[2])####TO DO check it

        if(self.device == 'cpu'):
            d_ceiling = self.resize_crop(d_ceiling.squeeze(0).squeeze(0).detach().numpy(),fc_ration, layout_mask.shape[2])####TO DO check it
            d_ceiling += f_dist.detach().numpy()
        else:
            d_ceiling = self.resize_crop(d_ceiling.squeeze(0).squeeze(0).detach().cpu().numpy(),fc_ration, layout_mask.shape[2])
            d_ceiling += f_dist.detach().cpu().numpy()
        
        if(self.device == 'cpu'):    
            heightmap = layout_mask.squeeze(1).squeeze(0) * d_ceiling
        else:
            heightmap = layout_mask.squeeze(1).squeeze(0) * torch.Tensor(d_ceiling).to(self.device)

        return heightmap 

    def resize_crop(self,img, scale, size):
    
            re_size = int(img.shape[0]*scale)

            if(re_size>0):
                img = cv2.resize(img, (re_size, re_size), cv2.INTER_AREA)

            if size <= re_size:
                pd = int((re_size-size)/2)
                img = img[pd:pd+size,pd:pd+size]
            else:
                new = np.zeros((size,size))
                pd = int((size-re_size)/2)
                new[pd:pd+re_size,pd:pd+re_size] = img[:,:]
                img = new

            return img    
  

def config_s3d():
    a = argparse.ArgumentParser(description='Generate nadir rooms map from Structured3D-like datasets')
    a.add_argument('--data_root', default='./data/s3d_floor', type=str, help='path to raw Structured3D_panorama folder')
    a.add_argument('--output', default='./results/s3d_nadirmaps', type=str, help='path to output folder')
    a.add_argument('--device', default='cuda', type=str, help='processor device')
    a.add_argument('--encode_heightmap', default=False, type=bool, help='encode ceiling heightmap')
    a.add_argument('--pth', default='./nadirshape/ckpt/DEMO_RUNS/s3d_depth/best_valid.pth', type=str, help='d2l pth')###n
    a.add_argument('--data_type', default='s3d', type=str, help='dataset s3d or zind')
    a.add_argument('--save_metric', default=True, type=bool, help='save metric information')
    a.add_argument('--split_file', default='', type=str, help='train/val/test splitting')###nb only for zind    
    
    args = a.parse_args()
    return args


def merge_nadir_shapes(args):
    data_root = args.data_root
        
    scenes = os.listdir(data_root)

    instance_id = 0

    ######annotations support
    ### prepare
    outFolder = args.output
    if not os.path.exists(outFolder):
        os.mkdir(outFolder)

    annotation_outFolder = os.path.join(outFolder, 'annotations')
    if not os.path.exists(annotation_outFolder):
        os.mkdir(annotation_outFolder)
            
    train_img_folder = os.path.join(outFolder, 'train')
    val_img_folder = os.path.join(outFolder, 'val')
    test_img_folder = os.path.join(outFolder, 'test')

    for img_folder in [train_img_folder, val_img_folder, test_img_folder]:
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

    coco_train_json_path = os.path.join(annotation_outFolder, 'train.json')
    coco_val_json_path = os.path.join(annotation_outFolder, 'val.json')
    coco_test_json_path = os.path.join(annotation_outFolder, 'test.json')

    coco_train_dict = {"images":[],"annotations":[],"categories":[],"metric":[]}
    coco_val_dict = {"images":[],"annotations":[],"categories":[],"metric":[]}
    coco_test_dict = {"images":[],"annotations":[],"categories":[],"metric":[]}

    for key, value in type2id.items():
        type_dict = {"supercategory": "room", "id": value, "name": key}
        coco_train_dict["categories"].append(type_dict)
        coco_val_dict["categories"].append(type_dict)
        coco_test_dict["categories"].append(type_dict)

    # Loading trained model
    net = load_trained_model(NadirShapeNet, args.pth).to(args.device)
    net.eval()

    fp_fov = net.fov
        
    d2l = D2L(gpu=True, H = 512, W = 1024, fp_fov = fp_fov)####new fov        

    for scene in tqdm(scenes):
        #####
        scene_path = os.path.join(data_root, scene)

        print('processing scene',scene)

        rtc = NadirTransformsCreator(scene_path, net, d2l, device = args.device, data_type=args.data_type,
                                     encode_heightmap = args.encode_heightmap)
         
        ####saved nadir map size
        wo = 256
        ho = 256                                            
               
        ###########GT starts                    
        scene_id = scene.split('_')[-1]

        if (int(scene_id) in invalid_scenes_ids):
            print('skip {}'.format(scene))
            ##continue
        else:           
            
            density, metric_normalization_dict = rtc.generate_floorplan_map(out_w=wo, out_h=ho, get_normalization = True)

            ### rescale raw annotations
            if(args.data_type=='zind'):
                ##scale_zind_annotations(scene_path)  ####CHECK IT 
                #NB. assuming annotations are already scaled to metric at rtc init time             
                normalized_annos = normalize_annotations(scene_path, metric_normalization_dict,json_name="annotation_3dm.json")
            else:
                normalized_annos = normalize_annotations(scene_path, metric_normalization_dict,json_name="annotation_3d.json")

            ### prepare coco dict
            img_id = int(scene_id)
            img_dict = {}
            img_dict["file_name"] = scene_id + '.png'
            img_dict["id"] = img_id
            img_dict["width"] = wo
            img_dict["height"] = ho
                        
            ### parse annotations
            polys = parse_floor_plan_polys(normalized_annos)#
            
            polygons_list = generate_coco_dict(normalized_annos, polys, instance_id, img_id, ignore_types=['outwall'])

            instance_id += len(polygons_list)
            
            if(args.save_metric):
                metric_dict = {}

                metric_normalization_dict["min_coords"] /= 1000.0 ###as meters
                metric_normalization_dict["max_coords"] /= 1000.0 

                metric_dict["min_coords"] = metric_normalization_dict["min_coords"].tolist()
                metric_dict["max_coords"] = metric_normalization_dict["max_coords"].tolist()
                metric_dict["image_res"] = metric_normalization_dict["image_res"].tolist()
                metric_dict["map_res"] = metric_normalization_dict["map_res"]
                metric_dict["heights"] = metric_normalization_dict["heights"]
                
                                              
                      
            #
            if(args.data_type == 'zind'):
                print('opening',args.split_file)
                with open(args.split_file, "r") as f:
                        split_json = json.load(f)

                        train_list = split_json['train']
                        val_list = split_json['val']
                        test_list = split_json['test']

                        print('ZIND train/val/test/tot buildings',len(train_list),len(val_list),len(test_list), len(train_list)+len(val_list)+len(test_list))                
                ### train
                if scene_id[:-1] in train_list:
                    ##print('DEBUG saving annotations', scene_id[:-1])
                    coco_train_dict["images"].append(img_dict)
                    coco_train_dict["annotations"] += polygons_list
                    export_density(density, train_img_folder, scene_id)

                    if(args.save_metric):
                        metric_path = os.path.join(train_img_folder, scene_id+'.json')                    

                ### val
                if scene_id[:-1] in val_list:
                    coco_val_dict["images"].append(img_dict)
                    coco_val_dict["annotations"] += polygons_list
                    export_density(density, val_img_folder, scene_id)

                    if(args.save_metric):
                        metric_path = os.path.join(val_img_folder, scene_id+'.json')                    

                ### test
                if scene_id[:-1] in test_list:
                    coco_test_dict["images"].append(img_dict)
                    coco_test_dict["annotations"] += polygons_list
                    export_density(density, test_img_folder, scene_id)

                    if(args.save_metric):
                        metric_path = os.path.join(test_img_folder, scene_id+'.json')                    
            else:             
                ### train
                if int(scene_id) < 3000:
                    coco_train_dict["images"].append(img_dict)
                    coco_train_dict["annotations"] += polygons_list
                    export_density(density, train_img_folder, scene_id)
                
                    if(args.save_metric):
                        metric_path = os.path.join(train_img_folder, scene_id+'.json')

                ### val
                elif int(scene_id) >= 3000 and int(scene_id) < 3250:
                    coco_val_dict["images"].append(img_dict)
                    coco_val_dict["annotations"] += polygons_list
                    export_density(density, val_img_folder, scene_id)
                
                    if(args.save_metric):
                        metric_path = os.path.join(val_img_folder, scene_id+'.json')

                ### test
                else:
                    coco_test_dict["images"].append(img_dict)
                    coco_test_dict["annotations"] += polygons_list
                    export_density(density, test_img_folder, scene_id)
                
                    if(args.save_metric):
                        metric_path = os.path.join(test_img_folder, scene_id+'.json')
                                                     

            with open(coco_train_json_path, 'w') as f:
                json.dump(coco_train_dict, f)
            with open(coco_val_json_path, 'w') as f:
                json.dump(coco_val_dict, f)
            with open(coco_test_json_path, 'w') as f:
                json.dump(coco_test_dict, f)
                
            if(args.save_metric):
                ##print('saving',metric_path)
                with open(metric_path, 'w') as f:
                    json.dump(metric_dict, f)


def main(args):

    ##os.environ['CUDA_VISIBLE_DEVICES'] = '3' 

    merge_nadir_shapes(args)

if __name__ == "__main__":
    main(config_s3d())