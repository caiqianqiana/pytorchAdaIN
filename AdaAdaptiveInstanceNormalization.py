import torch
import torch.nn as nn
import torch.nn.functional as F
def AdaptiveInstanceNormalization(input):
    eps=1e-5
    content = input['content']
    style = input['style']
    n = style.size(1)
    style=style.view(n,-1)
    targetStd,targetMean = torch.var_mean(style, 1)
    return F.instance_norm(content, weight=targetStd, bias=targetMean, eps=eps)