import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, down_time=False):
        super().__init__()
        # If down_time is True, we halve the temporal dimension (4->2, 2->1)
        pool = (2, 2, 2) if down_time else (1, 2, 2)
        self.net = nn.Sequential(
            nn.MaxPool3d(pool),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, up_time=False):
        super().__init__()
        k = (2, 2, 2) if up_time else (1, 2, 2)
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=k, stride=k)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        dT = x2.size(2) - x1.size(2)
        dH = x2.size(3) - x1.size(3)
        dW = x2.size(4) - x1.size(4)

        x1 = F.pad(x1, [dW // 2, dW - dW // 2,
                        dH // 2, dH - dH // 2,
                        dT // 2, dT - dT // 2])

        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, init_hid_dim=8, *args, **kwargs):
        super().__init__()
        
        # We ignore n_channels for the Conv layer because we move frames to Depth.
        # The actual input channel is 1 (the single pixel value).
        h = [init_hid_dim * (2 ** i) for i in range(5)]

        self.inc = DoubleConv(1, h[0])

        # Encoder: Designed for 4 frames (Depth=4)
        self.down1 = Down(h[0], h[1], down_time=True)  # D: 4 -> 2
        self.down2 = Down(h[1], h[2], down_time=True)  # D: 2 -> 1
        self.down3 = Down(h[2], h[3], down_time=False) # D stays 1
        self.down4 = Down(h[3], h[4], down_time=False) # D stays 1

        self.temporal = nn.Sequential(
            nn.Conv3d(h[4], h[4], (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(h[4]),
            nn.ReLU(inplace=True),
            nn.Conv3d(h[4], h[4], (3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(h[4]),
            nn.ReLU(inplace=True),
        )

        # Decoder: Mirroring the encoder
        self.up1 = Up(h[4], h[3], up_time=False)
        self.up2 = Up(h[3], h[2], up_time=False)
        self.up3 = Up(h[2], h[1], up_time=True)        # D: 1 -> 2
        self.up4 = Up(h[1], h[0], up_time=True)        # D: 2 -> 4

        self.outc = OutConv(h[0], n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input x is [Batch, 4, 1000, 500] (N, C, H, W)
        if x.dim() == 4:
            # Move frames to Depth: [Batch, 1, 4, 1000, 500] (N, C, D, H, W)
            x = x.unsqueeze(1) 

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.temporal(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x)
        # Collapse the Depth (temporal) dimension to get a 2D prediction map
        x = x.mean(dim=2) 
        return self.sigmoid(x)

# class DoubleConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         # DoubleConv should just process the data, not reshape it
#         return self.net(x)

# class Down(nn.Module):
#     def __init__(self, in_ch, out_ch, down_time=False):
#         super().__init__()
#         pool = (2, 2, 2) if down_time else (1, 2, 2)
#         self.net = nn.Sequential(
#             nn.MaxPool3d(pool),
#             DoubleConv(in_ch, out_ch)
#         )

#     def forward(self, x):
#         return self.net(x)

# class Up(nn.Module):
#     def __init__(self, in_ch, out_ch, up_time=False):
#         super().__init__()
#         k = (2, 2, 2) if up_time else (1, 2, 2)
#         # up_time=True: doubles D, H, and W
#         # up_time=False: doubles H and W, keeps D same
#         self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=k, stride=k)
#         self.conv = DoubleConv(in_ch, out_ch)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
        
#         # Calculate padding to ensure skip connection (x2) matches upsampled (x1)
#         dT = x2.size(2) - x1.size(2)
#         dH = x2.size(3) - x1.size(3)
#         dW = x2.size(4) - x1.size(4)

#         x1 = F.pad(x1, [dW // 2, dW - dW // 2,
#                         dH // 2, dH - dH // 2,
#                         dT // 2, dT - dT // 2])

#         return self.conv(torch.cat([x2, x1], dim=1))

# class UNet3D(nn.Module):
#     def __init__(self, n_channels, n_classes, init_hid_dim=8):
#         super().__init__()
#         h = [init_hid_dim * (2 ** i) for i in range(5)]

#         self.inc = DoubleConv(1, h[0]) # n_channels is 1 because we move frames to Depth

#         # Encoder: Reduce D from 4 -> 2 -> 1
#         self.down1 = Down(h[0], h[1], down_time=True)  # D: 4 -> 2
#         self.down2 = Down(h[1], h[2], down_time=True)  # D: 2 -> 1
#         self.down3 = Down(h[2], h[3], down_time=False) # D: 1 -> 1
#         self.down4 = Down(h[3], h[4], down_time=False) # D: 1 -> 1

#         self.temporal = nn.Sequential(
#             nn.Conv3d(h[4], h[4], (3, 1, 1), padding=(1, 0, 0), bias=False),
#             nn.BatchNorm3d(h[4]),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(h[4], h[4], (3, 1, 1), padding=(1, 0, 0), bias=False),
#             nn.BatchNorm3d(h[4]),
#             nn.ReLU(inplace=True),
#         )

#         # Decoder: Mirror the Encoder
#         self.up1 = Up(h[4], h[3], up_time=False)       # D: 1 -> 1
#         self.up2 = Up(h[3], h[2], up_time=False)       # D: 1 -> 1
#         self.up3 = Up(h[2], h[1], up_time=True)        # D: 1 -> 2
#         self.up4 = Up(h[1], h[0], up_time=True)        # D: 2 -> 4

#         self.outc = nn.Conv3d(h[0], n_classes, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # 1. FIX INPUT DIMENSIONS
#         # If input is (Batch, 4, 1000, 500)
#         if x.dim() == 4:
#             x = x.unsqueeze(1) # (B, 1, 4, 1000, 500) 
#             # Now C=1 and Depth=4. This is what a 3D UNet expects.
        
#         # 2. ENCODER
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)

#         # 3. BOTTLENECK
#         x5 = self.temporal(x5)

#         # 4. DECODER
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)

#         # 5. OUTPUT
#         x = self.outc(x)
#         # Collapse the time dimension at the end to get one 2D prediction map
#         x = x.mean(dim=2) 
#         return self.sigmoid(x)


# class DoubleConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         print(f"DoubleConv input shape: {x.shape}")
#         if x.dim() == 4:
#                 x = x.unsqueeze(2) 
#                 # Now x is [16, 4, 1, 1000, 500] (N, C, D, H, W)
            
#         # Swap C and D so that Frames are in the Depth dimension
#         # [16, 4, 1, 1000, 500] -> [16, 1, 4, 1000, 500]
#         x = x.transpose(1, 2)

#         x1 = self.inc(x)
#         return self.net(x1)


# class Down(nn.Module):
#     def __init__(self, in_ch, out_ch, down_time=False):
#         super().__init__()
#         pool = (2, 2, 2) if down_time else (1, 2, 2)
#         self.net = nn.Sequential(
#             nn.MaxPool3d(pool),
#             DoubleConv(in_ch, out_ch)
#         )

#     def forward(self, x):
#         return self.net(x)


# class Up(nn.Module):
#     def __init__(self, in_ch, out_ch, up_time=False):
#         super().__init__()
#         k = (2, 2, 2) if up_time else (1, 2, 2)
#         self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, k, k)
#         self.conv = DoubleConv(in_ch, out_ch)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)

#         dT = x2.size(2) - x1.size(2)
#         dH = x2.size(3) - x1.size(3)
#         dW = x2.size(4) - x1.size(4)

#         x1 = F.pad(
#             x1,
#             [dW // 2, dW - dW // 2,
#              dH // 2, dH - dH // 2,
#              dT // 2, dT - dT // 2]
#         )

#         return self.conv(torch.cat([x2, x1], dim=1))


# class OutConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Conv3d(in_ch, out_ch, 1)

#     def forward(self, x):
#         return self.conv(x)


# class UNet3D(nn.Module):
#     def __init__(self, n_channels, n_classes, init_hid_dim=8, *args, **kwargs):
#         super().__init__()

#         h = [init_hid_dim * (2 ** i) for i in range(5)]

#         self.inc = DoubleConv(n_channels, h[0])

#         # self.down1 = Down(h[0], h[1])
#         # self.down2 = Down(h[1], h[2])
#         # self.down3 = Down(h[2], h[3], down_time=True)
#         # self.down4 = Down(h[3], h[4])

#         # --- Encoder ---
#         self.down1 = Down(h[0], h[1], down_time=True)  # D: 4 -> 2
#         self.down2 = Down(h[1], h[2], down_time=True)  # D: 2 -> 1
#         self.down3 = Down(h[2], h[3], down_time=False) # D stays 1
#         self.down4 = Down(h[3], h[4], down_time=False) # D stays 1

#         self.temporal = nn.Sequential(
#             nn.Conv3d(h[4], h[4], (3, 1, 1), padding=(1, 0, 0), bias=False),
#             nn.BatchNorm3d(h[4]),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(h[4], h[4], (3, 1, 1), padding=(1, 0, 0), bias=False),
#             nn.BatchNorm3d(h[4]),
#             nn.ReLU(inplace=True),
#         )

#         # self.up1 = Up(h[4], h[3])
#         # self.up2 = Up(h[3], h[2], up_time=True)
#         # self.up3 = Up(h[2], h[1])
#         # self.up4 = Up(h[1], h[0])

#         # --- Decoder (Mirroring the encoder) ---
#         self.up1 = Up(h[4], h[3], up_time=False)
#         self.up2 = Up(h[3], h[2], up_time=False)
#         self.up3 = Up(h[2], h[1], up_time=True)        # D: 1 -> 2
#         self.up4 = Up(h[1], h[0], up_time=True)        # D: 2 -> 4

#         self.outc = OutConv(h[0], n_classes)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         if x.dim() == 4:
#             x = x.unsqueeze(2)

#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)

#         x5 = self.temporal(x5)

#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)

#         x = self.outc(x)
#         x = x.mean(dim=2)
#         return self.sigmoid(x)
