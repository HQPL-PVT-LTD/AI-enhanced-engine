"""One-off: convert Real-ESRGAN weights (.pth) to ONNX (needs torch + onnx).
Usage: python tools/export_onnx.py <fast|quality> in.pth out.onnx
  fast    realesr-general-x4v3 https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
  quality RealESRGAN_x4plus    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    def __init__(self, nf=64, nconv=32, up=4):
        super().__init__()
        self.up = up
        body = [nn.Conv2d(3, nf, 3, 1, 1), nn.PReLU(nf)]
        for _ in range(nconv):
            body += [nn.Conv2d(nf, nf, 3, 1, 1), nn.PReLU(nf)]
        body.append(nn.Conv2d(nf, 3 * up * up, 3, 1, 1))
        self.body = nn.ModuleList(body)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = F.pixel_shuffle(out, self.up)
        return out + F.interpolate(x, scale_factor=self.up, mode="nearest")


class RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.act(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = RDB(nf, gc), RDB(nf, gc), RDB(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):  # RealESRGAN_x4plus
    def __init__(self, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.act(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.act(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.act(self.conv_hr(feat)))


net = SRVGGNetCompact() if sys.argv[1] == "fast" else RRDBNet()
sd = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
net.load_state_dict(sd.get("params_ema", sd.get("params", sd)))
net.eval()
torch.onnx.export(net, torch.rand(1, 3, 64, 64), sys.argv[3], opset_version=17, dynamo=False,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "H", 3: "W"}})
