from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch as tc

device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings("ignore")


# ##################################################################################

# ================================= FusionUNetFNO =================================

# ##################################################################################

# ================================= FusionUNetFNO =================================
# Spectral Convolution
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = 1 / (in_channels * out_channels)

        self.weights = nn.Parameter(
            self.scale * tc.randn(
                in_channels, out_channels, modes1, 2) )

    def compl_mul1d(self, input, weights):
        # (B, in_c, x) * (in_c, out_c, x)
        return tc.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        B, _, T = x.shape
        x_ft = tc.fft.rfft(x, dim=-1)

        out_ft = tc.zeros(
            B, self.out_channels,
            T // 2 + 1,
            dtype=tc.cfloat,
            device=x.device
        )

        weights = tc.view_as_complex(self.weights)

        out_ft[:, :, :self.modes1] = self.compl_mul1d(
            x_ft[:, :, :self.modes1],
            weights
        )

        x = tc.fft.irfft(out_ft, n=T, dim=-1)

        return x


# Conv Block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),

            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch)
        )

        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        y = self.block(x)

        return F.gelu(y + self.skip(x))


# Hybrid Spectral Block
class HybridSpectralBlock(nn.Module):
    def __init__(self, in_ch, out_ch, modes=32 ):
        super().__init__()

        # Local branch
        self.local = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),

            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch)
        )

        # Spectral branch
        self.spectral = SpectralConv1d( in_ch, out_ch, modes1=modes )
        self.spec_norm = nn.GroupNorm(8, out_ch)

        # Learnable fusion
        self.gate = nn.Sequential(
            nn.Conv1d(out_ch * 2, out_ch, 1),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 1),
            nn.Sigmoid()
        )

        # Output mixing
        self.mix = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 1),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )

        # Residual
        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        local_feat = self.local(x)

        spectral_feat = self.spectral(x)
        spectral_feat = self.spec_norm(spectral_feat)
        spectral_feat = F.gelu(spectral_feat)

        fusion = tc.cat( [local_feat, spectral_feat], dim=1)

        alpha = self.gate(fusion)
        y = ( alpha * local_feat + (1.0 - alpha) * spectral_feat)
        y = self.mix(y)

        return F.gelu(y + self.skip(x))


# Downsample Block
class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_spectral=False, modes=32):
        super().__init__()

        if use_spectral:
            self.block = HybridSpectralBlock(in_ch, out_ch, modes=modes)
        else:
            self.block = ConvBlock(in_ch, out_ch)

        self.down = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )

    def forward(self, x):
        feat = self.block(x)
        down = self.down(feat)

        return feat, down


# Upsample Block
class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_spectral=False, modes=32):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),

            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1 ),

            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )

        if use_spectral:
            self.block = HybridSpectralBlock( out_ch + skip_ch, out_ch, modes=modes)
        else:
            self.block = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
        
        x = tc.cat([x, skip], dim=1)
        x = self.block(x)

        return x


class FusionUNetFNO(nn.Module):
    def __init__( self, input_channels=3, base_ch=16, modes=32 ):
        super().__init__()
        # Encoder
        self.enc1 = DownsampleBlock(input_channels, base_ch, use_spectral=False)
        self.enc2 = DownsampleBlock(base_ch, base_ch * 2, use_spectral=True, modes=modes)
        self.enc3 = DownsampleBlock(base_ch * 2, base_ch * 4, use_spectral=True, modes=modes)

        # Bottleneck
        self.bottleneck = HybridSpectralBlock(base_ch * 4, base_ch * 4, modes=modes)

        # Decoder
        self.dec3 = UpsampleBlock(base_ch * 4, base_ch * 4, base_ch * 2, use_spectral=True, modes=modes)
        self.dec2 = UpsampleBlock(base_ch * 2, base_ch * 2, base_ch, use_spectral=True, modes=modes)
        self.dec1 = UpsampleBlock(base_ch, base_ch, base_ch, use_spectral=False)

        # Refinement
        self.refine = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, 3, padding=1 ),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),

            nn.Conv1d( base_ch, base_ch // 2, 3, padding=1 ),
            nn.GroupNorm(4, base_ch // 2),
            nn.GELU()
        )

        # Output Heads
        self.outP = nn.Conv1d(base_ch // 2, 1, 1)
        self.outS = nn.Conv1d(base_ch // 2, 1, 1)

        self.apply(self._init_weights)

    # Weight Init
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(
                m.weight, nonlinearity='relu' )

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, SpectralConv1d):
            nn.init.normal_(m.weights[..., 0],
                mean=0.0,std=0.02 )

            nn.init.normal_(m.weights[..., 1],
                mean=0.0, std=0.02 )

    def forward(self, x):
        # Encoder
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        # Refinement
        x = self.refine(x)

        # Output Heads
        xP = self.outP(x)
        xS = self.outS(x)
        out = tc.cat([xP, xS], dim=1)

        return out
    



class SismoDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


def create_phase_labels(length, indices, device, sigma=10):
    batch_size = indices.shape[0]
    time = tc.arange(length, device=device).unsqueeze(0)  # [1, T]

    # Expand to [B, T]
    time = time.expand(batch_size, length).to(device)

    p_centers = indices[:, 0].reshape(-1, 1)
    s_centers = indices[:, 1].reshape(-1, 1)
    s_centers[s_centers > length] = 0

    # Create masks for "no phase"
    p_mask = (p_centers == 0)
    s_mask = (s_centers == 0)

    # P labels
    p_labels = tc.exp(-0.5 * ((time - p_centers) / sigma) ** 2).to(device)
    p_labels = p_labels / p_labels.max(dim=1, keepdim=True).values
    p_labels = tc.nan_to_num(p_labels, nan=0.0, posinf=0.0, neginf=0.0)
    p_labels = tc.clip(p_labels, 0, 1)
    p_labels[p_mask.expand_as(p_labels)] = 0.0

    # S labels
    s_labels = tc.exp(-0.5 * ((time - s_centers) / sigma) ** 2)
    s_labels = s_labels / s_labels.max(dim=1, keepdim=True).values
    s_labels = tc.nan_to_num(s_labels, nan=0.0, posinf=0.0, neginf=0.0)
    s_labels = tc.clip(s_labels, 0, 1)
    s_labels[s_mask.expand_as(s_labels)] = 0.0

    return tc.stack((p_labels, s_labels), dim=1).to(device)


def add_colored_noise(trace, low_freq=0.1, high_freq=1, dt=1/100):
    n = trace.shape[-1]
    freqs = tc.fft.fftfreq(n, d=dt).to(trace.device)

    noise = tc.randn_like(trace)
    noise_f = tc.fft.fft(noise)

    mask = (freqs.abs() >= low_freq) & (freqs.abs() <= high_freq)
    noise_f = noise_f * mask

    noise = tc.fft.ifft(noise_f).real
    return trace + noise


def mix_batch_events_torch(data, picks, device, length=6000, mix_prob=0.4, num_events_range=(1, 2),
                           noise_prob=0, max_shift=2000):
    data = data.to(device)
    picks = picks.to(device)

    N, _, T = data.shape
    new_data = []
    new_picks = []

    if mix_prob == 0:
        new_data = data.to(device)
        picks = create_phase_labels(length, tc.tensor(picks.to(device), device=device), device=device )
        return new_data, picks
    
    for _ in range(N):
        if tc.rand(1, device=device).item() < mix_prob:
            num_events = tc.randint(num_events_range[0], num_events_range[1] + 1, (1,), device=device).item()
            num_events = min(num_events, N)
            event_ids = tc.randperm(N, device=device)[:num_events]
            shifts = tc.randint(0, max_shift, (num_events,), device=device) + 800

            for j, tup in enumerate(zip(event_ids, shifts)):
                eid, shift = tup
                event = data[eid]
                event = event / (event.abs().max() + 1e-8)
                pick = picks[eid] + int(shift.item())
                if j == 0:
                    mixed_signal = event
                    orig = create_phase_labels(length, tc.tensor(picks[eid], device=device).unsqueeze(0), device=device )
                    mixed_picks = orig.squeeze(0)

                start = int(pick[0])
                end = int(pick[1].item() + 500)

                if pick[0] < T:
                    mixed_signal[:,start:end] += event[:,start:end]
                    pick = create_phase_labels(length, tc.tensor(pick, device=device).unsqueeze(0), device=device )
                    mixed_picks += pick.squeeze(0)
                    mixed_picks = tc.clip(mixed_picks, 0, 1)
                
            new_data.append(mixed_signal)
            new_picks.append(mixed_picks)

    try:
        if len(new_data) > 0 and len(new_picks) > 0:
            picks = create_phase_labels(length, tc.tensor(picks, device=device), device=device )
            new_data = tc.stack(new_data, dim=0).to(device)
            new_data = tc.cat((data, new_data), dim=0).to(device)
            new_picks = tc.stack(new_picks, dim=0).to(device)  
            new_picks = tc.cat((picks, new_picks), dim=0).to(device)
        else:
            new_data = data.to(device)
            new_picks = create_phase_labels(length, tc.tensor(picks.to(device), device=device), device=device )
    except:
        print( new_data )

    if tc.rand(1, device=device).item() < noise_prob:
        new_data = add_colored_noise(new_data)

    return new_data, new_picks