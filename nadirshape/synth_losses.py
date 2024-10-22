import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

###NEW
from layout.slicenet_gated_model_scalable import GatedSliceNet,FastSliceNet
from layout.misc import tools

from layout.geometry.render import *

from layout.geometry.panorama import transform_depthmap

from layout.misc import criteria
from layout.misc import ssim

from layout.misc.sobel import EdgeDetector
from torchvision.transforms.functional import rgb_to_grayscale

from layout.misc.atlanta_transform import E2P

import math
from torch.autograd import Variable

from layout.misc.layout import *

import numpy as np
from torch.autograd import Variable

EPS = 1e-6


def l1_loss(self, f1, f2, mask = 1):
    return torch.mean(torch.abs(f1 - f2)*mask)

'''
#TODO: Generative grayscale implementation 
def edge_loss(self, fake_edge, real_edge):
    grey_image = self.grey
    grey_image = F.interpolate(grey_image, size = fake_edge.size()[2:])
    fake_edge = torch.cat([fake_edge, grey_image], dim = 1)
    pred_fake, features_edge1 = self.D(fake_edge)
    return self.adversarial_loss(pred_fake, True, False)
'''
def style_loss_feats(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        _, c, w, h = A_feat.size()
        A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
        B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
        A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
        B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
        loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
    return loss_value

def TV_loss(x):
    h_x = x.size(2)
    w_x = x.size(3)
    h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
    w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
    return h_tv + w_tv 

def preceptual_loss_feats(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        loss_value += torch.mean(torch.abs(A_feat - B_feat))
        
class AdversarialLoss(nn.Module):

    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        #self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    # def __call__(self, x, y, VGG_feats):
    #     # Compute features
    #     x_vgg, y_vgg = VGG_feats(x), VGG_feats(y)

    #     # Compute loss
    #     style_loss = 0.0
    #     style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
    #     style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
    #     style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
    #     style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

    #     return style_loss


    def __call__(self, x_vgg, y_vgg):
        # Compute loss


        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss

class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        #self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y, VGG_feats)->torch.Tensor:
        # Compute features
        x_vgg, y_vgg = VGG_feats(x), VGG_feats(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss


    def __call__(self, x_vgg, y_vgg)->torch.Tensor:
        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss

class ContexualLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_vgg, y_vgg):
        #https://arxiv.org/pdf/1803.02077.pdf
        contexual_loss = 0
        content_loss = contextual_loss.functional.contextual_loss(x_vgg["relu4_2"],y_vgg["relu4_2"], band_width = 0.1)


class DistributedVGG19(torch.nn.Module):
    def __init__(self, proc_device = 'cpu', out_device = 'cuda'):
        super(DistributedVGG19, self).__init__()

        self.proc_device = proc_device
        self.out_device = out_device 

        features = models.vgg19(pretrained=True).features.to(self.proc_device)

        #normalization based on Structured3D mean/std
        self.register_buffer(
            name='vgg_mean',
            tensor=torch.tensor(
                [[[0.6160]], [[0.5899]], [[0.5502]]], requires_grad=False).to(self.proc_device)
        )
        self.register_buffer(
            name='vgg_std',
            tensor=torch.tensor(
                [[[0.1508]], [[0.1572]], [[0.1660]]], requires_grad=False).to(self.proc_device)
        )

        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):

        x = x.to(self.proc_device)

        #normalization
        x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())##.to(self.proc_device)

        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1.to(self.out_device),
            'relu1_2': relu1_2.to(self.out_device),

            'relu2_1': relu2_1.to(self.out_device),
            'relu2_2': relu2_2.to(self.out_device),

            'relu3_1': relu3_1.to(self.out_device),
            'relu3_2': relu3_2.to(self.out_device),
            'relu3_3': relu3_3.to(self.out_device),
            'relu3_4': relu3_4.to(self.out_device),

            'relu4_1': relu4_1.to(self.out_device),
            'relu4_2': relu4_2.to(self.out_device),
            'relu4_3': relu4_3.to(self.out_device),
            'relu4_4': relu4_4.to(self.out_device),

            'relu5_1': relu5_1.to(self.out_device),
            'relu5_2': relu5_2.to(self.out_device),
            'relu5_3': relu5_3.to(self.out_device),
            'relu5_4': relu5_4.to(self.out_device),
        }
        return out

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features

        #normalization based on Structured3D mean/std
        self.register_buffer(
            name='vgg_mean',
            tensor=torch.tensor(
                [[[0.6160]], [[0.5899]], [[0.5502]]], requires_grad=False)
        )
        self.register_buffer(
            name='vgg_std',
            tensor=torch.tensor(
                [[[0.1508]], [[0.1572]], [[0.1660]]], requires_grad=False)
        )

        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):

        #normalization
        x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())

        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out

def boundedLoss(
    regressed_coords    : torch.tensor, #b,h,w,c
    minimum, maximum, mask
):
    loss = torch.nn.MSELoss()
    zeros = torch.zeros_like(regressed_coords)
    ones = zeros + 1

    bigger = regressed_coords > maximum
    loss_bigger = torch.where(bigger, loss(regressed_coords, ones), zeros)

    smaller = regressed_coords < minimum
    loss_smaller = torch.where(smaller, loss(regressed_coords, -ones), zeros)


    if mask is None:
        total_loss = (loss_bigger + loss_smaller).mean()
    else:
        total_loss = torch.where(mask.byte(), loss_bigger + loss_smaller, zeros).sum() / mask.sum()


    return total_loss


##semantic losses

def get_IoU(outputs : torch.Tensor, labels : torch.Tensor):
    outputs = outputs.int()
    labels = labels.int()
    # Taken from: https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    intersection = (outputs & labels).float().sum((-1, -2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((-1, -2))  # Will be zero if both are 0

    iou = (intersection + EPS) / (union + EPS)  # We smooth our devision to avoid 0/0

    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    # return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch
    return iou.mean(dim = (-1,-2)).sum()

def get_segmentation_masks(seg_pred):
    soft_sem = torch.softmax(seg_pred, dim = 1) #####TO DO - here semantic is given by clutter mask
    soft_sem = torch.argmax(soft_sem, dim=1, keepdim=True)
    soft_sem = torch.clamp(soft_sem, min=0, max=1)
    masks = torch.zeros_like(seg_pred).to(seg_pred.device)
    masks.scatter_(1, soft_sem, 1)

    return masks

class FeatureMatchingLoss(nn.Module):
    def __init__(self, criterion='l1'):
        super(FeatureMatchingLoss, self).__init__()
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError('Criterion %s is not recognized' % criterion)

    def forward(self, fake_features, real_features):
        """
           fake_features (list of lists): Discriminator features of fake images.
           real_features (list of lists): Discriminator features of real images.
        Returns:
        (tensor): Loss value.
        """
        num_d = len(fake_features)
        dis_weight = 1.0 / num_d
        loss = fake_features[0][0].new_tensor(0)
        for i in range(num_d):
            for j in range(len(fake_features[i])):
                tmp_loss = self.criterion(fake_features[i][j],
                                          real_features[i][j].detach())
                loss += dis_weight * tmp_loss
        return loss


#####NEW latent geometric losses

class GatedSliceNet_Latent(torch.nn.Module):
    def __init__(self, device, estimator = None):
        super(GatedSliceNet_Latent, self).__init__()

        self.using_gatednet = False #####LEGACY - for compatibility

        if(estimator == None):
            d_pth='ckpt/resnet18_B8_s3d_512x1024_pure_rgb_mask_one_no_inter/best_valid.pth' ####N
            self.depth_estimator = tools.load_trained_model(GatedSliceNet, d_pth).to(device)
            self.using_gatednet = True

            print("Freezing segmentation network's weights\n")
            for param in self.depth_estimator.parameters():
                param.requires_grad = False
        else:
            self.depth_estimator = estimator                        
        

    def forward(self,img):
        if(self.using_gatednet):
            b, c, h, w = img.size() 
            x_spr = torch.zeros(b, 1, h, w).to(img.device)###FIXMEEE
        
        if(self.using_gatednet):        
            output = self.depth_estimator.get_latent_feature(img, x_spr) ####BxDxW
        else:
            output = self.depth_estimator.get_latent_feature(img) ####BxDxW
                           
        return output

class GatedSliceNet_Loss(torch.nn.Module):
    def __init__(self, device):
        super(GatedSliceNet_Loss, self).__init__()
        
        d_pth='ckpt/resnet18_B8_s3d_512x1024_pure_rgb_mask_one_no_inter/best_valid.pth' ####N
        self.depth_estimator = tools.load_trained_model(GatedSliceNet, d_pth).to(device)
        
        print("Freezing segmentation network's weights\n")

        for param in self.depth_estimator.parameters():
            param.requires_grad = False
                            
        
        
    def forward(self, ti_pred, ti_gt, d_pred = None):
        b, c, h, w = ti_gt.size() 
        x_spr = torch.zeros(b, 1, h, w).to(ti_gt.device)

        #####DIRECT part
        loss = 0.0
                
        d1 = self.depth_estimator(ti_pred,x_spr)
        d2 = self.depth_estimator(ti_gt,x_spr)
               
        d_loss = criteria.inverse_huber_loss(d1,d2)
        
        #####LATENT part
        l1 = self.depth_estimator.get_latent_feature(ti_pred,x_spr)
        l2 = self.depth_estimator.get_latent_feature(ti_gt,x_spr)

        result = []

        result.append(d_loss)
        
        if(d_pred != None):
            dir_loss = criteria.inverse_huber_loss(d_pred,d2)####Bxhxw
            result.append(dir_loss)

        result.append(l1)
        result.append(l2)
                                                   
        return result

class SliceNet_Loss(torch.nn.Module):
    def __init__(self, proc_device = 'cuda', out_device = 'cuda', d_pth ='ckpt/S3D_combo_depth_atl_layout_gated_max_min_loss/best_valid.pth'):
        super(SliceNet_Loss, self).__init__()

        self.d2l = D2L(gpu=False)

        self.proc_device = proc_device
        self.out_device = out_device 

        print('processing device', self.proc_device,'out device', self.out_device)
        
                    
        self.layout_estimator = tools.load_trained_model(FastSliceNet, d_pth).to(self.proc_device)
        self.layout_estimator.eval()
                            
        print("SliceNet_Loss: Freezing segmentation network's weights\n")
            
        for param in self.layout_estimator.parameters():
            param.requires_grad = False
          
            
    def get_depth_layout(self, src_img, x_t, get_layout = False):
        ####FIXME - handle W!1024

        b,c,h,w =  src_img.size()

        if(w!=1024):
            src_img1 = F.interpolate(src_img, size=(512, 1024), mode='nearest')
            d, prob = self.layout_estimator(src_img1.to(self.proc_device))
        else:
            d, prob = self.layout_estimator(src_img.to(self.proc_device))
        
        if(get_layout):
            x_dle = self.d2l.get_translated_layout_edges(d, prob, x_t)
            if(w!=1024):
                d = F.interpolate(d.unsqueeze(1), size=(h, w), mode='nearest')

                d = d.squeeze(1)

            return d.to(self.out_device),x_dle.to(self.out_device)
        else:
            if(w!=1024):
                d = F.interpolate(d.unsqueeze(1), size=(h, w), mode='nearest')
                d = d.squeeze(1)

            return d.to(self.out_device)        
                
    def forward(self, ti_pred, ti_gt, d_pred=None):
        #####DIRECT part
        loss = 0.0

        b,c,h,w =  ti_pred.size()

        if(w!=1024):
            ti_pred1 = F.interpolate(ti_pred, size=(512, 1024), mode='nearest')
            ti_gt1 = F.interpolate(ti_gt, size=(512, 1024), mode='nearest')
            d1, prob1 = self.layout_estimator(ti_pred1.to(self.proc_device))
            d2, prob2 = self.layout_estimator(ti_gt1.to(self.proc_device))
            if(d_pred != None):
                d_pred1 = F.interpolate(d_pred.unsqueeze(1), size=(512, 1024), mode='nearest')
        else:
            d1, prob1 = self.layout_estimator(ti_pred.to(self.proc_device))
            d2, prob2 = self.layout_estimator(ti_gt.to(self.proc_device))                
        
        m1 = get_segmentation_masks(prob1)
        m2 = get_segmentation_masks(prob2)

        d_loss = criteria.inverse_huber_loss(d1,d2)
        m_loss = F.binary_cross_entropy_with_logits(m1,m2)

        #####LATENT part
        l1 = self.layout_estimator.get_latent_feature(ti_pred.to(self.proc_device))
        l2 = self.layout_estimator.get_latent_feature(ti_gt.to(self.proc_device))
          

        result = []

        result.append(d_loss.to(self.out_device))
        result.append(m_loss.to(self.out_device))

        if(d_pred != None):        
            if(w!=1024):
                dir_loss = criteria.inverse_huber_loss(d_pred1.to(self.proc_device),d2)####Bxhxw
            else:
                dir_loss = criteria.inverse_huber_loss(d_pred.to(self.proc_device),d2)####Bxhxw

            result.append(dir_loss.to(self.out_device))

        result.append(l1.to(self.out_device))
        result.append(l2.to(self.out_device))
                                                   
        return result

class GeometricPerceptualLoss(nn.Module):
    ###
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(GeometricPerceptualLoss, self).__init__()
        
        self.criterion = torch.nn.L1Loss()
        self.weights = weights ###NOT USED
            
    def __call__(self, x_slice, y_slice, multi_feats = False)->torch.Tensor:
        content_loss = 0.0

        if(multi_feats):
            for xs, ys in zip(x_slice, y_slice):
                content_loss += self.criterion(xs, ys)
        else:
            content_loss += self.criterion(x_slice, y_slice)
        
        return content_loss

class GeometricStyleLoss(nn.Module):
    ###
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(GeometricStyleLoss, self).__init__()
        
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def compute_gram(self, x):
        b, ch, s = x.size()
        f = x ### b,ch,s
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (s * ch)

        return G
            
    def __call__(self, x_slice, y_slice, multi_feats = False)->torch.Tensor:
        content_loss = 0.0

        if(multi_feats):
            for xs, ys in zip(x_slice, y_slice):
                content_loss += self.criterion(self.compute_gram(xs), self.compute_gram(ys))
        else:
            content_loss += self.criterion(self.compute_gram(x_slice), self.compute_gram(y_slice))
        
        return content_loss

    
class UnsupervisedPhotometricLoss(torch.nn.Module):
    def __init__(self):
        super(UnsupervisedPhotometricLoss, self).__init__()

        self.criterion1 = torch.nn.L1Loss()
        self.criterion2 = ssim.SSIM(window_size = 11)
                
        self.eta = 0.85
        
    def photometric_loss(self,x1,x2,m):
        return self.eta*torch.clamp( (1.0-self.criterion2(x2, m*x1)),0.0,1.0 ) + (1.0-self.eta)*self.criterion1(x2, m*x1)

    def inverse_huber_loss(self,target,output):
        absdiff = torch.abs(output-target)
        C = 0.2*torch.max(absdiff).item()
        return torch.mean(torch.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C) ))
        
    def forward(self, img0, depth0, img1, depth1, x_t):
        ####src_img, src_depth, trg_img_gt, trg_pred_depth
        loss = 0.0

        ###NB. default: img0, depth0, img1, x_t are GT - depth1 predicted candidate

        img10, mask10 = render(img0, depth0, x_t, max_depth=10.0, get_mask = True, masked_img = True, filter_iter = 1, masks_th = 0.9, use_tr_depth = True) ####splatted src img with src gt depth
        img01, mask01 = render(img1, depth1, -x_t, max_depth=10.0, get_mask = True, masked_img = True, filter_iter = 1, masks_th = 0.9, use_tr_depth = True) ####splatted trg_img_gt with pred depth at t1 - back to src position
                
        depth01 = transform_depthmap(depth0, x_t)               
        depth10, maskd10 = render(depth01, depth0, x_t, max_depth=10.0, get_mask = True, masked_img = True, filter_iter = 1, masks_th = 0.9, use_tr_depth = True)####splatted src_depth 

        ##DEBUG
        loss10 = self.criterion1(img10, mask10*img1) #### GT projection error - splatted src image vs. masked trg_img_gt
        ##loss01 = self.criterion1(img01, mask01*img0) ####src img from backprojection using predicted depth        
        ##print('splatting loss',loss10)
        
        loss01 = self.photometric_loss(img01,img0, mask01)####src img from backprojection using predicted depth  FIXED
        
        ##DEBUG
        ##loss10 = self.photometric_loss(img1,img10,mask10)#GT projection error 

                
        loss_d10 = self.inverse_huber_loss(depth10, maskd10*depth1) ####force infilling
                            
               
        return (loss01),loss_d10

class EdgeLoss(torch.nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()

        self.ed = EdgeDetector(use_cuda=True)
        
        self.th_edges = 0.1

    def edgemap(self,img):
        ig = rgb_to_grayscale(img, num_output_channels= 1)##rgb2gray(img)
                
        edge_im  = self.ed(ig)
                
        if(self.th_edges != -1):
            edge_im = torch.where(edge_im > self.th_edges, 1.0, 0.)

        return edge_im  

    def forward(self, ti_pred, e_gt):
        e_pred = self.edgemap(ti_pred)

        print(e_pred.shape,e_gt.shape)

        return F.binary_cross_entropy_with_logits(e_pred,e_gt)

class E2PLoss(torch.nn.Module):
    def __init__(self, w = 1024, h=512, split_loss = False):
        super(E2PLoss, self).__init__()

        self.e2p = E2P(equ_size=(h, w), out_dim=1024, fov=160, radius=1, gpu=True)

        self.criterion1 = torch.nn.L1Loss()
        ##self.criterion2 = ssim.SSIM(window_size = 11)

        self.split_loss = split_loss
      
      
    def forward(self, ti_pred, ti_gt):
        [p_up, p_down] = self.e2p(ti_pred)
        [gt_up, gt_down] = self.e2p(ti_gt)

        l1_up = self.criterion1(p_up,gt_up)
        l1_dn = self.criterion1(p_down,gt_down)

        if(self.split_loss):
            return l1_up,l1_dn
        else:
            return (l1_up+l1_dn)

class max_min_Loss(torch.nn.Module):
    def __init__(self, device, H=512, W = 1024):
        super(max_min_Loss, self).__init__()

        self.xz_sph = self.atlanta_sphere(H, W, device)

        self.criterion = nn.L1Loss()
            
    def atlanta_sphere(self,H,W,device):
        ####build xyz sphere coordinates
        P = np.zeros(shape =(2, H, W),dtype=np.float32) 

        for i in range(H):
            theta = -np.pi * (float(i)/float(H-1)-0.5)
            for j in range(W):
                phi = np.pi * (2.0*float(j)/float(W-1)-1.0)
               
                ##P[0,i,j] = math.cos(phi)*math.cos(theta)
                P[1,i,j] = math.sin(theta) ####Z
                ##P[1,i,j] = math.sin(phi)*math.cos(theta)

       
        P = Variable(torch.FloatTensor(P)).to(device)
        
        ##xx = P[0] * P[0]
        zz = P[1] * P[1]
        
        D = torch.sqrt(zz)
                
        D = D.unsqueeze(0) ###to batched shape ##xyz: 1x3xhxw
                    
        return D

    def euclidean_to_planar_depth(self, d, xz_sph):  
        ##input depth: 1xhxw  

        DP = xz_sph.to(d.device) * d  ## (1x3xhxw) * (1xhxw)                

        return  DP 

    def max_min_depth(self, src_depth):
        src_depth_plan = self.euclidean_to_planar_depth(src_depth.squeeze(0), self.xz_sph)##
                
        B,H,W = src_depth_plan.size()

        up_depth, bottom_depth = torch.split(src_depth_plan, H//2, dim=1)

        MN = []
        
        for i in range(B):
            max = torch.max(up_depth[i:i+1])
            min = torch.max(bottom_depth[i:i+1])

            max_min = torch.zeros(2).to(src_depth_plan.device)
            max_min[0] = max
            max_min[1] = min

            MN.append(max_min.unsqueeze(0))
           
        MN = torch.cat(MN, dim=0)####Bx2
            
                
        return MN

    def forward(self, d_pred, d_gt):
        pred_mn = self.max_min_depth(d_pred)
        gt_mn   = self.max_min_depth(d_gt)
                                
        return self.criterion(pred_mn, gt_mn)

class Occupancy_Loss(torch.nn.Module):
    def __init__(self, H=512, W = 1024, proc_device = 'cuda', out_device = 'cuda'):
        super(Occupancy_Loss, self).__init__()

        self.H = H 
        self.W = W 

        ##self.device = device     
        #
        self.proc_device = proc_device 
        self.out_device = out_device         

        self.xyz_sph_h = self.xyz_sphere_custom(ref_plan = 'horizontal').to(self.proc_device) 
        self.xyz_sph_v = self.xyz_sphere_custom(ref_plan = 'vertical').to(self.proc_device)
        self.xyz_sph_l = self.xyz_sphere_custom(ref_plan = 'lateral').to(self.proc_device)  

        print('DEBUG INIT', self.proc_device, self.out_device)                              

    def xyz_sphere_custom(self, ref_plan = 'horizontal'):####NB. phi and theta have different convention 
    ####build xyz sphere coordinates
        P = np.zeros(shape =(3, self.H, self.W),dtype=np.float32) 
        ##P = torch.zeros((3,H,W),dtype=torch.float32, device = device)        

        for i in range(self.H):
            theta = -np.pi * (float(i)/float(self.H-1)-0.5)
            for j in range(self.W):
                phi = np.pi * (2.0*float(j)/float(self.W-1)-1.0)

                if(ref_plan == 'horizontal'):
                    P[0,i,j] = math.cos(phi)*math.cos(theta)
                    P[1,i,j] = math.sin(phi)*math.cos(theta)
                    P[2,i,j] = math.sin(theta) ####Z                    
                                                       

                if(ref_plan == 'vertical'):
                    P[0,i,j] = math.cos(phi)*math.cos(theta)
                    P[1,i,j] = -math.sin(theta) ####Z
                    P[2,i,j] = math.sin(phi)*math.cos(theta)

                if(ref_plan == 'lateral'):
                    #P[0,i,j] = math.sin(theta) ####Z
                    #P[1,i,j] = math.sin(phi)*math.cos(theta) 
                    #P[2,i,j] = math.cos(phi)*math.cos(theta)

                    P[0,i,j] = math.sin(phi)*math.cos(theta)                    
                    P[1,i,j] = -math.sin(theta) ####Z
                    P[2,i,j] = math.cos(phi)*math.cos(theta)                   
                                                         
               
        #eps = 0.001
        #P += eps
                    
        # if use_gpu:
        #     P = Variable(torch.FloatTensor(P)).cuda()
        # else:
        #     P = Variable(torch.FloatTensor(P))

        P = Variable(torch.FloatTensor(P)).cuda()

        P = P.unsqueeze(0) ###to batched shape ##xyz: 1x3xhxw
                    
        return P

    def depth2density_gpu(self, x_depth, xyz_sph, height=512, width=512, abs_scale = 20000.0, unit_scale = 1000.0):###input depth: BxHxW
          
        ##x_depth *= unit_scale 
        #
        ##device = x_depth.device       

        ####NEW faster generator
        ##xyz_sph = self.xyz_sphere_custom(x_depth.shape[1], x_depth.shape[2], ref_plan = ref_plan).to(device)  
     
        b_size = x_depth.shape[0]

        BOM = []

        x_depth = x_depth.to(self.proc_device)
        ##xyz_sph.to(self.proc_device)                
   
        ##print('DEBUG depth2density_gpu',x_depth.device, xyz_sph.device)        
    
        for i in range(b_size):     
            ps = (x_depth[i:i+1] * xyz_sph).squeeze(0).reshape(3,-1).permute(1,0)                 

            ps *= unit_scale  
                    
               
            coordinates = ps[:, :2] + abs_scale/2.0 ####2D       
            coordinates /= abs_scale
         
            density = torch.zeros((height, width),dtype=torch.float32).to(ps.device)
            image_res = torch.asarray(torch.tensor([height, width]),dtype=torch.int64).to(ps.device)

            coordinates = torch.round(coordinates*image_res[None])
            coordinates = torch.minimum(torch.maximum(coordinates, torch.zeros_like(image_res)), image_res - 1)      
                    
            unique_coordinates, counts = torch.unique(coordinates, return_counts=True, dim=0)

            th = 1e2                    
    
            counts = torch.minimum(counts, torch.tensor(th))
            unique_coordinates = unique_coordinates.long()
            
            density[unique_coordinates[:, 1], unique_coordinates[:, 0]] = counts.to(dtype=torch.float32)##.to(device)

            ##print(' dmax', torch.max(density))        
        
            density = density / torch.max(density)

            density = density.unsqueeze(0)
            
            BOM.append(density) 

        b_density = torch.cat(BOM, dim=0)####Bxhxw                                   

        return b_density       
                

    def forward(self, pred_depth, gt_depth):
        ##print('OM loss forward',pred_depth.device, gt_depth.device)
        
        pred_om_h = self.depth2density_gpu(pred_depth, self.xyz_sph_h, height=512, width=512)
        gt_om_h = self.depth2density_gpu(gt_depth, self.xyz_sph_h, height=512, width=512) 

        pred_om_v = self.depth2density_gpu(pred_depth, self.xyz_sph_v, height=512, width=512)
        gt_om_v = self.depth2density_gpu(gt_depth, self.xyz_sph_v, height=512, width=512)   

        pred_om_l = self.depth2density_gpu(pred_depth, self.xyz_sph_l, height=512, width=512)
        gt_om_l = self.depth2density_gpu(gt_depth, self.xyz_sph_l, height=512, width=512)
                   

        l_h = criteria.inverse_huber_loss(pred_om_h, gt_om_h).to(self.out_device) ####Bxhxw
        l_v = criteria.inverse_huber_loss(pred_om_v, gt_om_v).to(self.out_device) ####Bxhxw  
        l_l = criteria.inverse_huber_loss(pred_om_l, gt_om_l).to(self.out_device) ####Bxhxw 

        ##print(l_h,l_v,l_l)                              
                                
        return (l_h+l_v+l_l)    

class DepthDiffRenderingLoss(torch.nn.Module):
    def __init__(self):
        super(DepthDiffRenderingLoss, self).__init__()

        self.criterion1 = torch.nn.L1Loss()
        
    def inverse_huber_loss(self,target,output):
        absdiff = torch.abs(output-target)
        C = 0.2*torch.max(absdiff).item()
        return torch.mean(torch.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C) ))
        
    def forward(self, depth0, depth1):
        ####src_img, src_depth, trg_img_gt, trg_pred_depth
        ##loss = 0.0

        b,h,w = depth0.size()

        XT = []

        for i in range(b):
            x_t = torch.FloatTensor(np.random.random_sample((1,1,3)))                

            x_t[:,:,0] = x_t[:,:,1] - 0.5
            x_t[:,:,1] = x_t[:,:,1] - 0.5
            x_t[:,:,2] = x_t[:,:,2] - 0.5

            XT.append(x_t)

        XT = torch.cat(XT, dim=0)
                
        w1, sw1 = get_weights(depth0.unsqueeze(1), XT.to(depth0.device), max_depth=10.0)
        w2, sw2 = get_weights(depth1.unsqueeze(1), XT.to(depth1.device), max_depth=10.0)
                        
        loss1 = self.criterion1(w1, w2)
        loss2 = self.criterion1(sw1, sw2)
         
        loss = loss2
               
        return loss
            

## .d2l.max_min_depth(torch.FloatTensor(depth).unsqueeze(0))

if __name__ == '__main__':
    device = torch.device('cuda')
    ##edge_loss = EdgeLoss().to(device)
    ##e2p_loss = E2PLoss(w=1024, h=512).to(device)
    ##max_min_loss = max_min_Loss(device)

    h,w = 256,512##512,1024

    B = 8

    ti_ = torch.randn(B,3,h,w).to(device)
    ti_gt = torch.randn(B,3,h,w).to(device)
    ti_d = torch.randn(B,h,w).to(device)
    ti_d_gt = torch.randn(B,h,w).to(device)

    x_t = torch.randn(1,3).to(device)
    
    #sl_loss = SliceNet_Loss().to(device)
        
    #d,w,p,l1,l2 = sl_loss(ti_, ti_gt, ti_d)

    #print(l1.shape,l2.shape)

    #dp = sl_loss.get_depth_layout(ti_,x_t)

    #print(dp.shape)

    ##splat_loss = DepthDiffRenderingLoss().to(device)

    ##sl = splat_loss(ti_d, ti_d_gt)

    ti_d = torch.randn(B,h,w).to(device)
    ti_d_gt = torch.randn(B,h,w).to(device)
    om_loss = Occupancy_Loss().to(device) 
    ol = om_loss(ti_d.to(device), ti_d_gt.to(device))

    print(ol)

