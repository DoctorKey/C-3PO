import os
import numpy as np
import torch
from PIL import Image
import glob
from torchvision.transforms import functional as F

import dataset.transforms as T 
from dataset.dataset import CDDataset
import dataset.path_config as Data_path


def Dict_indexing():
    Dict = {}
    Dict['background']={'index':0,'subnames':[]}
    Dict['column']={'index':1,'subnames':['pillar','pilar']} #sero
    Dict['pipe']={'index':2,'subnames':['tube']}
    Dict['wall']={'index':3,'subnames':['tunnel']}
    Dict['beam']={'index':4,'subnames':[]} # garo
    Dict['floor']={'index':5,'subnames':['slam','ground','road','walk','floorpanel']}
    Dict['frame']={'index':6,'subnames':['scafolding','scaffolding','scaffold','formwork','pole','support']}
    Dict['fence']={'index':7,'subnames':['fencning']}

    Dict['wire']={'index':8,'subnames':['wirecylinder']}
    Dict['cable']={'index':9,'subnames':[]}
    Dict['window']={'index':10,'subnames':['glass_panel']}
    Dict['railing']={'index':11,'subnames':[]}
    Dict['rail']={'index':12,'subnames':[]}
    Dict['ceiling']={'index':13,'subnames':['roof']}
    Dict['stair']={'index':14,'subnames':[]}
    Dict['duct']={'index':15,'subnames':['vent','ventilation']}
    Dict['gril']={'index':16,'subnames':['grid']}  # bunker's platform

    Dict['lamp']={'index':17,'subnames':['light']} # GOOD
    Dict['trash']={'index':18,'subnames':['debris','book','paper']} # GOOD
    Dict['shelf']={'index':19,'subnames':['drawer','rack','locker','cabinet']}
    Dict['door']={'index':20,'subnames':['gate']} #GOOD
    Dict['barrel']={'index':21,'subnames':['barel','drum','tank']} # GOOD
    Dict['sign']={'index':22,'subnames':['signcver']} # GOOD
    Dict['box']={'index':23,'subnames':['paperbox','bin','cube','crateplastic']} # Good
    Dict['bag']={'index':24,'subnames':[]} # GOOD
    Dict['electric_box']={'index':25,'subnames':['fusebox','switchboard','electricalsupply',
                                             'electric_panel','powerbox','control_panel']} # GOOD
    Dict['vehicle']={'index':26,'subnames':['truck','trailer','transporter','forklift']}
    Dict['ladder']={'index':27,'subnames':[]} # GOOD
    Dict['canister']={'index':28,'subnames':['can','bottle','cylinder','keg']}
    Dict['extinguisher']={'index':29,'subnames':['fire_ex']} # GOOD
    Dict['pallet'] = {'index': 30, 'subnames': ['palete', 'palette']}  # GOOD
    Dict['hand_truck'] = {'index': 31, 'subnames': ['pumptruck','pallet_jack']}  # GOOD

    return Dict


class SegHelper:
    def __init__(self,opt=None,idx2color_path='../../backup/idx2color.txt',num_class=32):
        self.opt = opt
        self.num_classes = num_class
        self.idx2color_path = idx2color_path
        f = open(self.idx2color_path, 'r')
        self.idx2color = {k:[] for k in range(self.num_classes)}
        for j in range(256):
            line = f.readline()
            line = line.strip(' \n').strip('[').strip(']').strip(' ').split()
            line = [int(l) for l in line if l.isdigit()]
            self.idx2color[j] = line # color in rgb order

        self.color2idx = {tuple(v):k for k,v in self.idx2color.items()}
        name2idx = Dict_indexing()
        self.name2idx = {k: name2idx[k]['index'] for k in name2idx.keys()}
        self.idx2name = {v:k for k,v in self.name2idx.items()}
        self.idx2name_padding = {v:'BG' for v in range(self.num_classes,256)}
        self.idx2name.update(self.idx2name_padding)

    def unique(self,array):
        uniq, index = np.unique(array, return_index=True, axis=0)
        return uniq[index.argsort()]

    def extract_color_from_seg(self,img_seg):
        colors = img_seg.reshape(-1, img_seg.shape[-1]) # (H*W,3) # color channel in rgb order
        unique_colors = self.unique(colors) # (num_class_in_img,3)
        return unique_colors

    def extract_class_from_seg(self,img_seg):
        unique_colors = self.extract_color_from_seg(img_seg) # (num_class_in_img,3) # color channel in rgb order
        classes_idx = [self.color2idx[tuple(color.tolist())]for color in unique_colors]
        classes_str = [self.idx2name[idx] for idx in classes_idx]
        return classes_idx, classes_str

    def colormap2classmap(self,seg_array):
        seg_array_flattened = torch.LongTensor(seg_array.reshape(-1,3))
        seg_map_class_flattened = torch.zeros((seg_array.shape[0],seg_array.shape[1],1)).view(-1,1)
        for color, cls in self.color2idx.items():
            matching_indices = (seg_array_flattened == torch.LongTensor(color))
            matching_indices = (matching_indices.sum(dim=1)==3)
            seg_map_class_flattened[matching_indices] = cls
        seg_map_class = seg_map_class_flattened.view(seg_array.shape[0],seg_array.shape[1],1)
        # return CPU
        seg_map_class = seg_map_class.squeeze().long()
        return seg_map_class

    def classmap2colormap(self,seg_map_class):
        seg_map_class_flattened = seg_map_class.view(-1,1)
        seg_map_color_flattened = torch.zeros(seg_map_class.shape[0]*seg_map_class.shape[1],3).cuda().long()
        for cls, color in self.idx2color.items():
            matching_indices = (seg_map_class_flattened == torch.LongTensor([cls]).cuda())
            seg_map_color_flattened[matching_indices.view(-1)] = torch.LongTensor(color).cuda()
        seg_map_color_flattened = seg_map_color_flattened.view(seg_map_class.shape[0],seg_map_class.shape[1],3)
        seg_map_color_flattened = seg_map_color_flattened.cpu().permute(2,0,1)
        return seg_map_color_flattened

    def split_SemAndChange(self,seg_map_class):
        seg_map_change_class = seg_map_class//50
        seg_map_semantic_class = torch.fmod(seg_map_class,50)
        return seg_map_semantic_class, seg_map_change_class


class ChangeSim(CDDataset):
    def __init__(self, ROOT='', split='train', num_classes=2, seg=None, transforms=None, revert_transforms=None):
        """
        ChangeSim Dataloader
        Please download ChangeSim Dataset in https://github.com/SAMMiCA/ChangeSim
        Args:
            num_classes (int): Number of target change detection class
                               5 for multi-class change detection
                               2 for binary change detection (default: 5)
            set (str): 'train' or 'test' (defalut: 'train')
        """
        super(ChangeSim, self).__init__(ROOT, transforms)
        self.num_classes = num_classes
        self.set = split
        train_list = ['Warehouse_0', 'Warehouse_1', 'Warehouse_2', 'Warehouse_3', 'Warehouse_4', 'Warehouse_5']
        test_list = ['Warehouse_6', 'Warehouse_7', 'Warehouse_8', 'Warehouse_9']
        self.image_total_files = []
        if split == 'train':
            for map in train_list:
                self.image_total_files += glob.glob(ROOT + '/Query/Query_Seq_Train/' + map + '/Seq_0/rgb/*.png')
                self.image_total_files += glob.glob(ROOT + '/Query/Query_Seq_Train/' + map + '/Seq_1/rgb/*.png')
        elif split == 'test':
            for map in test_list:
                self.image_total_files += glob.glob(ROOT + '/Query/Query_Seq_Test/' + map + '/Seq_0/rgb/*.png')
                self.image_total_files += glob.glob(ROOT + '/Query/Query_Seq_Test/' + map + '/Seq_1/rgb/*.png')

        self.seg = seg
        self._transforms = transforms
        self._revert_transforms = revert_transforms

    def __len__(self):
        return len(self.image_total_files)

    def get_raw(self, index):
        test_rgb_path = self.image_total_files[index]
        ref_rgb_path = test_rgb_path.replace('rgb', 't0/rgb')
        change_segmentation_path = test_rgb_path.replace('rgb', 'change_segmentation')

        img_t0 = self._pil_loader(ref_rgb_path)
        img_t1 = self._pil_loader(test_rgb_path)
        imgs = [img_t0, img_t1]

        mask = self._pil_loader(change_segmentation_path)
        if self.num_classes == 2:
            mask = mask.convert("L")
        return imgs, mask

    def get_mask_ratio(self):
        if self.num_classes == 2:
            return [0.0846, 0.9154]

    def get_pil(self, imgs, mask, pred=None):
        assert self._revert_transforms is not None
        t0, t1 = self._revert_transforms(imgs.cpu())
        w, h = t0.size
        output = Image.new('RGB', (w * 2, h * 2))
        output.paste(t0)
        output.paste(t1, (w, 0))
        if self.num_classes == 5:
            mask = self.seg.classmap2colormap(mask.cuda())
            pred = self.seg.classmap2colormap(pred.cuda())
        mask = F.to_pil_image(mask.cpu().float())
        pred = F.to_pil_image(pred.cpu().float())
        output.paste(mask, (0, h))
        output.paste(pred, (w, h))
        return output


def get_ChangeSim(args, train=True, num_class=2):
    input_size = args.input_size
    raw_root = Data_path.get_dataset_path('ChangeSim')
    size_dict = {
        256: (256, 256),
        512: (512, 512),
    }
    assert input_size in size_dict, "input_size: {}".format(size_dict.keys())
    input_size = size_dict[input_size]
    mode = 'train' if train else 'test'

    if num_class == 2 or num_class == 5:
        seg_class_num = 5
    else:
        seg_class_num = 32
    seg = SegHelper(idx2color_path=os.path.join(raw_root, 'idx2color.txt'), num_class=seg_class_num)

    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)

    print("{} Aug:".format(mode))
    augs = []
    if train:
        augs.append(T.Resize(input_size))
        augs.append(T.RandomHorizontalFlip(args.randomflip))
        #augs.append(T.ColorJitter(0.4, 0.4, 0.4, 0.25))
    else:
        augs.append(T.Resize(input_size))

    if num_class == 2:
        augs.append(T.ToTensor())
        augs.append(T.Normalize(mean=mean, std=std))
        augs.append(T.ConcatImages())
    elif num_class == 5:
        augs.append(T.ToTensor(seg.colormap2classmap))
        augs.append(T.Normalize(mean=mean, std=std))
        augs.append(T.ConcatImages())
    else:
        augs.append(T.ToTensor(seg.colormap2classmap))
        augs.append(T.Normalize(mean=mean, std=std))

    transforms = T.Compose(augs)

    if num_class == 2 or num_class == 5:
        revert_transforms = T.Compose([
            T.SplitImages(),
            T.RevertNormalize(mean=mean, std=std),
            T.ToPILImage()
        ])
    else:
        revert_transforms = T.Compose([
            T.RevertNormalize(mean=mean, std=std),
            T.ToPILImage()
        ])
    return raw_root, mode, seg, transforms, revert_transforms

def get_ChangeSim_Binary(args, train=True):
    raw_root, mode, seg, transforms, revert_transforms = get_ChangeSim(args, train, 2)
    dataset = ChangeSim(raw_root, mode, num_classes=2, seg=seg,
        transforms=transforms, revert_transforms=revert_transforms)
    print("ChangeSim Binary {}: {}".format(mode, len(dataset)))
    return dataset

def get_ChangeSim_Multi(args, train=True):
    raw_root, mode, seg, transforms, revert_transforms = get_ChangeSim(args, train, 5)
    dataset = ChangeSim(raw_root, mode, num_classes=5, seg=seg,
        transforms=transforms, revert_transforms=revert_transforms)
    print("ChangeSim Multi {}: {}".format(mode, len(dataset)))
    return dataset

class ChangeSim_Semantic(CDDataset):
    def __init__(self, ROOT='', split='train', seg=None, transforms=None, revert_transforms=None):
        """
        ChangeSim Dataloader
        Please download ChangeSim Dataset in https://github.com/SAMMiCA/ChangeSim
        Args:
            set (str): 'train' or 'test' (defalut: 'train')
        """
        super(ChangeSim_Semantic, self).__init__(ROOT, transforms)
        self.set = split
        self.num_classes = 32
        self.class_mask = [1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        train_list = ['Warehouse_0', 'Warehouse_1', 'Warehouse_2', 'Warehouse_3', 'Warehouse_4', 'Warehouse_5']
        test_list = ['Warehouse_6', 'Warehouse_7', 'Warehouse_8', 'Warehouse_9']
        self.image_total_files = []
        if split == 'train':
            for map in train_list:
                self.image_total_files += glob.glob(ROOT + '/Query/Query_Seq_Train/' + map + '/Seq_0/rgb/*.png')
                self.image_total_files += glob.glob(ROOT + '/Query/Query_Seq_Train/' + map + '/Seq_1/rgb/*.png')
                self.image_total_files += glob.glob(ROOT + '/Reference/Ref_Seq_Train/' + map + '/Seq_0/rgb/*.png')
                self.image_total_files += glob.glob(ROOT + '/Reference/Ref_Seq_Train/' + map + '/Seq_1/rgb/*.png')
        elif split == 'test':
            for map in test_list:
                self.image_total_files += glob.glob(ROOT + '/Query/Query_Seq_Test/' + map + '/Seq_0/rgb/*.png')
                self.image_total_files += glob.glob(ROOT + '/Query/Query_Seq_Test/' + map + '/Seq_1/rgb/*.png')
                self.image_total_files += glob.glob(ROOT + '/Reference/Ref_Seq_Test/' + map + '/Seq_0/rgb/*.png')
                self.image_total_files += glob.glob(ROOT + '/Reference/Ref_Seq_Test/' + map + '/Seq_1/rgb/*.png')
        #self.image_total_files = self.image_total_files[:1000]
        self.seg = seg
        self._transforms = transforms
        self._revert_transforms = revert_transforms

    def __len__(self):
        return len(self.image_total_files)

    def get_raw(self, index):
        test_rgb_path = self.image_total_files[index]
        semantic_segmentation_path = test_rgb_path.replace('rgb', 'semantic_segmentation')
        img = self._pil_loader(test_rgb_path)
        mask = self._pil_loader(semantic_segmentation_path)
        return img, mask

    def get_pil(self, imgs, mask, pred=None):
        assert self._revert_transforms is not None
        t0 = self._revert_transforms(imgs.cpu())
        w, h = t0.size
        output = Image.new('RGB', (w * 3, h))
        output.paste(t0)
        mask = self.seg.classmap2colormap(mask.cuda())
        pred = self.seg.classmap2colormap(pred.cuda())
        mask = F.to_pil_image(mask.cpu().float())
        pred = F.to_pil_image(pred.cpu().float())
        output.paste(mask, (w, 0))
        output.paste(pred, (2*w, 0))
        return output


def get_ChangeSim_Semantic(args, train=True):
    raw_root, mode, seg, transforms, revert_transforms = get_ChangeSim(args, train, 32)
    dataset = ChangeSim_Semantic(raw_root, mode, seg=seg,
        transforms=transforms, revert_transforms=revert_transforms)
    print("ChangeSim_Semantic {}: {}".format(mode, len(dataset)))
    """
    exist = [False] * dataset.num_classes
    for i in range(len(dataset)):
        _, mask = dataset[i]
        all_count = mask.numel()
        for c in range(dataset.num_classes):
            target = (mask == c).long()
            all_count -= target.sum()
            if exist[c] is False and target.sum() > 0:
                exist[c] = True
        assert all_count == 0, "{} {}".format(i, all_count)
        if i % 100 == 0:
            print(i)
    print(exist)
    """
    return dataset
