
from model.FCN import resnet18_mtf_msf_fcn, biresnet18_mtf_msf_fcn, resnet18_msf_fcn

from model.DeepLabV3 import resnet18_mtf_msf_deeplabv3, mtf_resnet18_msf_deeplabv3, resnet18_msf_mtf_deeplabv3, resnet18_msf_deeplabv3_mtf
from model.DeepLabV3 import mobilenetv2_mtf_msf_deeplabv3
from model.DeepLabV3 import vgg16bn_mtf_msf_deeplabv3
from model.DeepLabV3 import resnet50_mtf_msf_deeplabv3
from model.DeepLabV3 import resnet18_msf_deeplabv3


model_dict = {

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
}
