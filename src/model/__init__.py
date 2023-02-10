from model.TANet import dr_tanet_refine_resnet18, dr_tanet_resnet18, tanet_refine_resnet18, tanet_resnet18
from model.cscdnet import cdresnet, cscdnet
from model.unet import FC_EF
from model.siamunet_conc import FC_Siam_conc
from model.siamunet_diff import FC_Siam_diff
from model.changenet import changenet


from model.FCN import resnet18_mtf_msf_fcn, biresnet18_mtf_msf_fcn, resnet18_msf_fcn

from model.DeepLabV3 import resnet18_mtf_msf_deeplabv3, mtf_resnet18_msf_deeplabv3, resnet18_msf_mtf_deeplabv3, resnet18_msf_deeplabv3_mtf
from model.DeepLabV3 import mobilenetv2_mtf_msf_deeplabv3
from model.DeepLabV3 import vgg16bn_mtf_msf_deeplabv3
from model.DeepLabV3 import resnet50_mtf_msf_deeplabv3
from model.DeepLabV3 import swinT_mtf_msf_deeplabv3
from model.DeepLabV3 import resnet18_msf_deeplabv3


model_dict = {
    'dr_tanet_refine_resnet18': dr_tanet_refine_resnet18,
    'dr_tanet_resnet18': dr_tanet_resnet18,
    'tanet_refine_resnet18': tanet_refine_resnet18,
    'tanet_resnet18': tanet_resnet18,

    'fc_ef': FC_EF,
    'fc_siam_conc': FC_Siam_conc,
    'fc_siam_diff': FC_Siam_diff,

    'changenet': changenet,

    'cdresnet': cdresnet,
    'cscdnet': cscdnet,

    'resnet18_mtf_msf_fcn': resnet18_mtf_msf_fcn,
    'biresnet18_mtf_msf_fcn': biresnet18_mtf_msf_fcn,
    'resnet18_msf_fcn': resnet18_msf_fcn,

    'resnet18_msf_deeplabv3': resnet18_msf_deeplabv3,
    'resnet18_mtf_msf_deeplabv3': resnet18_mtf_msf_deeplabv3,
    'mtf_resnet18_msf_deeplabv3': mtf_resnet18_msf_deeplabv3,
    'resnet18_msf_mtf_deeplabv3': resnet18_msf_mtf_deeplabv3,
    'resnet18_msf_deeplabv3_mtf': resnet18_msf_deeplabv3_mtf,

    'mobilenetv2_mtf_msf_deeplabv3': mobilenetv2_mtf_msf_deeplabv3,
    'vgg16bn_mtf_msf_deeplabv3': vgg16bn_mtf_msf_deeplabv3,
    'resnet50_mtf_msf_deeplabv3': resnet50_mtf_msf_deeplabv3,
    'swinT_mtf_msf_deeplabv3': swinT_mtf_msf_deeplabv3,

}
