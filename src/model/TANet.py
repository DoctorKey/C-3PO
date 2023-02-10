import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init


class Temporal_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 groups=1, bias=False, refinement=False):
        super(Temporal_Attention, self).__init__()
        self.outc = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.refinement = refinement

        print('Attention Layer-kernel size:{0},stride:{1},padding:{2},groups:{3}...'.format(self.kernel_size,self.stride,self.padding,self.groups))
        if self.refinement:
            print("Attention with refinement...")

        assert self.outc % self.groups == 0, 'out_channels should be divided by groups.'

        self.w_q = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.w_k = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.w_v = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)


        #relative positional encoding...
        self.rel_h = nn.Parameter(torch.randn(self.outc // 2, 1, 1, self.kernel_size, 1), requires_grad = True)
        self.rel_w = nn.Parameter(torch.randn(self.outc // 2, 1, 1, 1, self.kernel_size), requires_grad = True)
        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


        init.kaiming_normal_(self.w_q.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.w_k.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.w_v.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, feature_map):

        fm_t0, fm_t1 = torch.split(feature_map, feature_map.size()[1]//2, 1)
        assert fm_t0.size() == fm_t1.size(), 'The size of feature maps of image t0 and t1 should be same.'

        batch, _, h, w = fm_t0.size()


        padded_fm_t0 = F.pad(fm_t0, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.w_q(fm_t1)
        k_out = self.w_k(padded_fm_t0)
        v_out = self.w_v(padded_fm_t0)

        if self.refinement:

            padding = self.kernel_size
            padded_fm_col = F.pad(fm_t0, [0, 0, padding, padding])
            padded_fm_row = F.pad(fm_t0, [padding, padding, 0, 0])
            k_out_col = self.w_k(padded_fm_col)
            k_out_row = self.w_k(padded_fm_row)
            v_out_col = self.w_v(padded_fm_col)
            v_out_row = self.w_v(padded_fm_row)

            k_out_col = k_out_col.unfold(2, self.kernel_size * 2 + 1, self.stride)
            k_out_row = k_out_row.unfold(3, self.kernel_size * 2 + 1, self.stride)
            v_out_col = v_out_col.unfold(2, self.kernel_size * 2 + 1, self.stride)
            v_out_row = v_out_row.unfold(3, self.kernel_size * 2 + 1, self.stride)


        q_out_base = q_out.view(batch, self.groups, self.outc // self.groups, h, w, 1).repeat(1, 1, 1, 1, 1, self.kernel_size*self.kernel_size)
        q_out_ref = q_out.view(batch, self.groups, self.outc // self.groups, h, w, 1).repeat(1, 1, 1, 1, 1, self.kernel_size * 2 + 1)
        
        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.outc // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.outc // self.groups, h, w, -1)

        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.contiguous().view(batch, self.groups, self.outc // self.groups, h, w, -1)

        inter_out = (q_out_base * k_out).sum(dim=2)

        out = F.softmax(inter_out, dim=-1)
        out = torch.einsum('bnhwk,bnchwk -> bnchw', out, v_out).contiguous().view(batch, -1, h, w)

        if self.refinement:

            k_out_row = k_out_row.contiguous().view(batch, self.groups, self.outc // self.groups, h, w, -1)
            k_out_col = k_out_col.contiguous().view(batch, self.groups, self.outc // self.groups, h, w, -1)
            v_out_row = v_out_row.contiguous().view(batch, self.groups, self.outc // self.groups, h, w, -1)
            v_out_col = v_out_col.contiguous().view(batch, self.groups, self.outc // self.groups, h, w, -1)

            out_row = F.softmax((q_out_ref * k_out_row).sum(dim=2),dim=-1)
            out_col = F.softmax((q_out_ref * k_out_col).sum(dim=2),dim=-1)
            out += torch.einsum('bnhwk,bnchwk -> bnchw', out_row, v_out_row).contiguous().view(batch, -1, h, w)
            out += torch.einsum('bnhwk,bnchwk -> bnchw', out_col, v_out_col).contiguous().view(batch, -1, h, w)

        return out


upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2  # same conv
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding, bias=bias, dilation=dilation))


class Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3):
        super(Upsample, self).__init__()
        print(f'Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn)

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        skip_size = skip.size()[2:4]
        x = upsample(x, skip_size)
        x = x + skip
        x = self.blend_conv.forward(x)
        return x

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}

def conv3x3(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)

def block_function_factory(conv,norm,relu=None):
    def block_function(x):
        x = conv(x)
        if norm is not None:
            x = norm(x)
        if relu is not None:
            x = relu(x)
        return x
    return block_function

def do_efficient_fwd(block_f,x,efficient):
    if efficient and x.requires_grad:
        return cp.checkpoint(block_f,x)
    else:
        return block_f(x)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self,in_c,out_c,stride=1,downsample = None,efficient=True,use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(in_c,out_c,stride)
        self.bn1 = nn.BatchNorm2d(out_c) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_c,out_c)
        self.bn2 = nn.BatchNorm2d(out_c) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        block_f1 = block_function_factory(self.conv1,self.bn1,self.relu)
        block_f2 = block_function_factory(self.conv2,self.bn2)

        out = do_efficient_fwd(block_f1,x,self.efficient)
        out = do_efficient_fwd(block_f2,out,self.efficient)

        out = out + residual
        relu_out = self.relu(out)

        return relu_out,out

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True):
        super(Bottleneck, self).__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) if self.use_bn else None
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x

        bn_1 = block_function_factory(self.conv1, self.bn1, self.relu)
        bn_2 = block_function_factory(self.conv2, self.bn2, self.relu)
        bn_3 = block_function_factory(self.conv3, self.bn3)

        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)
        out = do_efficient_fwd(bn_3, out, self.efficient)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        relu_out = self.relu(out)

        return relu_out, out

class ResNet(nn.Module):

    def __init__(self, block, layers, efficient=False, use_bn=True, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.use_bn = use_bn
        self.efficient = efficient

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x:x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(self.inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(self.inplanes, planes, efficient=self.efficient, use_bn=self.use_bn)]

        return nn.Sequential(*layers)

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward(self, image):

        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer4)
        features += [skip]
        return features

class AttentionModule(nn.Module):

    def __init__(self, local_kernel_size = 1, stride = 1, padding = 0, groups = 1,
                 drtam = False, refinement = False, channels = [64,128,256,512]):
        super(AttentionModule, self).__init__()

        if not drtam:
            self.attention_layer1 = Temporal_Attention(channels[0], channels[0], local_kernel_size, stride, padding, groups, refinement=refinement)
            self.attention_layer2 = Temporal_Attention(channels[1], channels[1], local_kernel_size, stride, padding, groups, refinement=refinement)
            self.attention_layer3 = Temporal_Attention(channels[2], channels[2], local_kernel_size, stride, padding, groups, refinement=refinement)
            self.attention_layer4 = Temporal_Attention(channels[3], channels[3], local_kernel_size, stride, padding, groups, refinement=refinement)
        else:
            self.attention_layer1 = Temporal_Attention(channels[0], channels[0], 7, 1, 3, groups, refinement=refinement)
            self.attention_layer2 = Temporal_Attention(channels[1], channels[1], 5, 1, 2, groups, refinement=refinement)
            self.attention_layer3 = Temporal_Attention(channels[2], channels[2], 3, 1, 1, groups, refinement=refinement)
            self.attention_layer4 = Temporal_Attention(channels[3], channels[3], 1, 1, 0, groups, refinement=refinement)


        self.downsample1 = conv3x3(channels[0], channels[1], stride=2)
        self.downsample2 = conv3x3(channels[1]*2, channels[2], stride=2)
        self.downsample3 = conv3x3(channels[2]*2, channels[3], stride=2)

    def forward(self, features):

        features_t0, features_t1 = features[:4], features[4:]

        fm1 = torch.cat([features_t0[0],features_t1[0]], 1)
        attention1 = self.attention_layer1(fm1)
        fm2 = torch.cat([features_t0[1], features_t1[1]], 1)
        attention2 = self.attention_layer2(fm2)
        fm3 = torch.cat([features_t0[2], features_t1[2]], 1)
        attention3 = self.attention_layer3(fm3)
        fm4 = torch.cat([features_t0[3], features_t1[3]], 1)
        attention4 = self.attention_layer4(fm4)

        downsampled_attention1 = self.downsample1(attention1)
        cat_attention2 = torch.cat([downsampled_attention1,attention2], 1)
        downsampled_attention2 = self.downsample2(cat_attention2)
        cat_attention3 = torch.cat([downsampled_attention2,attention3], 1)
        downsampled_attention3 = self.downsample3(cat_attention3)
        final_attention_map = torch.cat([downsampled_attention3,attention4], 1)
        
        features_map = [final_attention_map,attention4,attention3,attention2,attention1]
        return features_map


class Decoder(nn.Module):
    
    def __init__(self,channels=[64,128,256,512]):
        super(Decoder, self).__init__()
        self.upsample1 = Upsample(num_maps_in=channels[3]*2, skip_maps_in=channels[3], num_maps_out=channels[3])
        self.upsample2 = Upsample(num_maps_in=channels[2]*2, skip_maps_in=channels[2], num_maps_out=channels[2])
        self.upsample3 = Upsample(num_maps_in=channels[1]*2, skip_maps_in=channels[1], num_maps_out=channels[1])
        self.upsample4 = Upsample(num_maps_in=channels[0]*2, skip_maps_in=channels[0], num_maps_out=channels[0])

    def forward(self, feutures_map):
        
        x = feutures_map[0]
        x = self.upsample1(x, feutures_map[1])
        x = self.upsample2(x, feutures_map[2])
        x = self.upsample3(x, feutures_map[3])
        x = self.upsample4(x, feutures_map[4])
        return x


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    channels = [64,128,256,512]
    return model,channels

def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    channels = [64, 128, 256, 512]
    return model,channels


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    channels = [256, 512, 1024, 2048]
    return model,channels

def get_encoder(arch,pretrained=True):
    if arch == 'resnet18':
        return resnet18(pretrained)
    elif arch == 'resnet34':
        return resnet34(pretrained)
    elif arch == 'resnet50':
        return resnet50(pretrained)
    else:
        print('Given the invalid architecture for ResNet...')
        exit(-1)

def get_attentionmodule(local_kernel_size = 1, stride = 1, padding = 0, groups = 1, drtam = False, refinement = False, channels=[64,128,256,512]):
    return AttentionModule(local_kernel_size=local_kernel_size,stride=stride, padding=padding, groups=groups,
                           drtam=drtam, refinement=refinement, channels=channels)
def get_decoder(channels=[64,128,256,512]):
    return Decoder(channels=channels)



class TANet(nn.Module):

    def __init__(self, encoder_arch, local_kernel_size, stride, padding, groups, drtam, refinement):
        super(TANet, self).__init__()
        self.encoder1, channels = get_encoder(encoder_arch,pretrained=True)
        self.encoder2, _ = get_encoder(encoder_arch,pretrained=True)
        self.attention_module = get_attentionmodule(local_kernel_size, stride, padding, groups, drtam, refinement, channels)
        self.decoder = get_decoder(channels=channels)
        self.classifier = nn.Conv2d(channels[0], 2, 1, padding=0, stride=1)
        self.bn = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img):

        img_t0,img_t1 = torch.split(img,3,1)
        features_t0 = self.encoder1(img_t0)
        features_t1 = self.encoder2(img_t1)
        features = features_t0 + features_t1
        features_map = self.attention_module(features)
        pred_ = self.decoder(features_map)
        pred_ = upsample(pred_,[pred_.size()[2]*2, pred_.size()[3]*2])
        pred_ = self.bn(pred_)
        pred_ = upsample(pred_,[pred_.size()[2]*2, pred_.size()[3]*2])
        pred_ = self.relu(pred_)
        pred = self.classifier(pred_)

        return pred


def dr_tanet_refine_resnet18(args):
    local_kernel_size = 1
    attn_stride = 1
    attn_padding = 0
    attn_groups = 4
    drtam = True
    refinement = True
    model = TANet('resnet18', local_kernel_size, attn_stride,
                    attn_padding, attn_groups, drtam, refinement)
    return model

def dr_tanet_resnet18(args):
    local_kernel_size = 1
    attn_stride = 1
    attn_padding = 0
    attn_groups = 4
    drtam = True
    refinement = False
    model = TANet('resnet18', local_kernel_size, attn_stride,
                    attn_padding, attn_groups, drtam, refinement)
    return model

def tanet_refine_resnet18(args):
    local_kernel_size = 1
    attn_stride = 1
    attn_padding = 0
    attn_groups = 4
    drtam = False
    refinement = True
    model = TANet('resnet18', local_kernel_size, attn_stride,
                    attn_padding, attn_groups, drtam, refinement)
    return model

def tanet_resnet18(args):
    local_kernel_size = 1
    attn_stride = 1
    attn_padding = 0
    attn_groups = 4
    drtam = False
    refinement = False
    model = TANet('resnet18', local_kernel_size, attn_stride,
                    attn_padding, attn_groups, drtam, refinement)
    return model