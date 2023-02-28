import torch.nn.functional as F

def cal_mean_std(x, eps=1e-5):
    B, C = x.size()[:2]
    x_mean = x.reshape(B, C, -1).mean(dim=2).reshape(B, C, 1, 1)
    x_std = x.reshape(B, C, -1).std(dim=2).reshape(B, C, 1, 1) + eps
    return x_mean, x_std

def content_loss(recover_feature, t):
    return F.mse_loss(recover_feature, t)

def style_loss(recover_mid_feature, style_mid_feature):
        loss = 0
        for c, s in zip(recover_mid_feature, style_mid_feature):
            c_mean, c_std = cal_mean_std(c)
            s_mean, s_std = cal_mean_std(s)
            loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)

        return loss