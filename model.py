import torch.nn as nn
import torchvision.models as models

from losses import cal_mean_std, content_loss, style_loss

def adain(c, s):
    c_mean, c_std = cal_mean_std(c)
    s_mean, s_std = cal_mean_std(s)
    return s_std * (c - c_mean) / c_std + s_mean

class Vgg19Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg19 = models.vgg19(pretrained=True).features
        self.stage1 = vgg19[:2]
        self.stage2 = vgg19[2:7]
        self.stage3 = vgg19[7:12]
        self.stage4 = vgg19[12:21]
        
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, last_feature=True):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        if last_feature:
            return x4
        else:
            return x1, x2, x3, x4

class RC(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, act=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((p, p, p, p))
        self.conv = nn.Conv2d(in_ch, out_ch, k)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(self.pad(x)))

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc1 = RC(512, 256, 3, 1)
        self.rc2 = RC(256, 256, 3, 1)
        self.rc3 = RC(256, 256, 3, 1)
        self.rc4 = RC(256, 256, 3, 1)
        self.rc5 = RC(256, 128, 3, 1)
        self.rc6 = RC(128, 128, 3, 1)
        self.rc7 = RC(128, 64, 3, 1)
        self.rc8 = RC(64, 64, 3, 1)
        self.rc9 = RC(64, 3, 3, 1, False)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, x):
        x = self.rc1(x)
        x = self.up(x)
        x = self.rc2(x)
        x = self.rc3(x)
        x = self.rc4(x)
        x = self.rc5(x)
        x = self.up(x)
        x = self.rc6(x)
        x = self.rc7(x)
        x = self.up(x)
        x = self.rc8(x)
        x = self.rc9(x)

        return x

class AdainStyleTransfom(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.vgg19 = Vgg19Encoder()
        self.decoder = Decoder()
        self.alpha = alpha

    def predict(self, content, style):
        c_feature = self.vgg19(content, last_feature=True)
        s_feature = self.vgg19(style, last_feature=True)

        t = adain(c_feature, s_feature)
        T = self.alpha * t + (1 - self.alpha) * c_feature

        recover = self.decoder(T)
        return recover
        
    def forward(self, content, style):
        c_feature = self.vgg19(content, last_feature=True)
        s_feature = self.vgg19(style, last_feature=True)

        t = adain(c_feature, s_feature)
        T = self.alpha * t + (1 - self.alpha) * c_feature

        recover = self.decoder(T)
        recover_feature = self.vgg19(recover, last_feature=True)
        recover_mid_feature = self.vgg19(recover, last_feature=False)
        style_mid_feature = self.vgg19(style, last_feature=False)

        loss_c = content_loss(recover_feature, t)
        loss_s = style_loss(recover_mid_feature, style_mid_feature)
        
        return loss_c, loss_s