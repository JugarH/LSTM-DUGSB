import os
import math
import random
import json
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Any

import numpy as np
import pickle as pkl
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch import Tensor
from torch.nn import Parameter
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Activation helpers from your code
σ = torch.sigmoid
sw = lambda x: x * σ(x)
φ_s = lambda x, Λ=10: (1 / Λ) * (Λ * sw(x + 1) - Λ * sw(x - 1)) - 1
ψ_s = lambda x, A=100, B=1.01: σ(A * (torch.abs(x) - B))

BASE_PATH = Path(__file__).parent
LOG_PATH = BASE_PATH / 'pth'; LOG_PATH.mkdir(exist_ok=True)

# ----------------------------
# to_ising
# ----------------------------
def to_ising_ext(H: Tensor, y: Tensor, nbps: int, lmbd: Tensor):
    """
    PyTorch implementation of to_ising_ext from your code.
    Inputs:
        H: [Nr, Nt] complex64 tensor
        y: [Nr] complex64 tensor
        nbps: number bits per symbol (e.g. 4 for 16QAM)
        lmbd: scalar Tensor or float
    Returns:
        J: [rb*N, rb*N] float32
        h: [rb*N, 1] float32
    """
    M = 2 ** nbps
    Nr, Nt = H.shape
    N = 2 * Nt
    rb = nbps // 2
    qam_var = 2 * (M - 1) / 3

    I = torch.eye(N, device=H.device)
    # build T: [N, rb*N]
    T = (2 ** (rb - 1 - torch.arange(rb, device=H.device)))[:, None, None] * I[None, ...]
    T = T.reshape(-1, N).T  # [N, rb*N]

    H_tilde = torch.vstack([
        torch.hstack([H.real, -H.imag]),
        torch.hstack([H.imag, H.real]),
    ])  # [2*Nr, 2*Nt]
    y_tilde = torch.cat([y.real, y.imag])  # [2*Nr]

    # LMMSE-like
    # lmbd may be tensor or float
    lmbd_val = lmbd if isinstance(lmbd, torch.Tensor) else torch.tensor([lmbd], device=H.device)
    # Use inverse safely
    U_λ = torch.linalg.inv(H_tilde @ H_tilde.T + (lmbd_val * torch.eye(2 * Nr, device=H.device))) / lmbd_val

    H_tilde_T = H_tilde @ T  # [2*Nr, rb*N] @? careful shapes: H_tilde [2Nr,2Nt], T [2Nt?] but we follow your earlier design
    # Note: in your code H_tilde @ T produced H_tilde_T; keep same arithmetic
    # J = - H_tilde_T.T @ H_tilde_T * (2.0 / qam_var)
    J = - H_tilde_T.T @ U_λ @ H_tilde_T * (2.0 / qam_var)
    J = J * (1.0 - torch.eye(J.shape[0], device=J.device))  # zero diagonal

    # compute h
    ones_rbN = torch.ones((J.shape[0], 1), device=H.device)
    ones_N = torch.ones((N, 1), device=H.device)
    z = (y_tilde.unsqueeze(1) - H_tilde_T @ ones_rbN + (math.sqrt(M) - 1) * (H_tilde @ ones_N)) / math.sqrt(qam_var)
    # h = 2.0 * H_tilde_T.T @ (z)  # shape [rb*N, 1]
    h = 2.0 * H_tilde_T.T @ (U_λ @ z)

    return J.to(torch.float32), h.to(torch.float32)

# ----------------------------
# calc_energy
# ----------------------------
def calc_energy(x: Tensor, J: Tensor, h: Tensor):
    """
    x: [dim, B]
    J: [dim, dim]
    h: [dim, 1] or None
    returns: energy [B]
    """
    sign = φ_s(x)
    # avoid zeros
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    s = sign.to(device)
    As = torch.matmul(J, s)  # [dim, B]
    if h is not None:
        energy = -0.5 * torch.sum(s * As, dim=0) - torch.sum(h * s, dim=0)
    else:
        energy = -0.5 * torch.sum(s * As, dim=0)
    return energy

# ----------------------------
# ber conversion: spins -> bits (numpy) and compute_ber
# ----------------------------
def spins_to_bits_numpy(solution: np.ndarray, nbps: int) -> np.ndarray:
    """
    Convert solution (Ising {-1,1}) to bits in constellation-style (0/1)
    solution: shape (rb*2*Nt, ) or (rb*2*Nt, 1)
    returns bits: [Nt, nbps] with values {0,1}
    """
    sol = solution.copy().astype(np.int32).reshape((-1,))
    # nbps -> rb
    rb = nbps // 2
    # shape to [rb, 2, Nt] — original code used [rb, c=2, Nt]
    # we don't know Nt explicitly; infer Nt = len(sol) // (2*rb)
    Nt = sol.shape[0] // (2 * rb)
    bits_hat = sol.reshape(rb, 2, Nt)  # [rb, 2, Nt]
    bits_hat = np.concatenate([bits_hat[:, 0], bits_hat[:, 1]], axis=0)  # [2*rb, Nt]
    bits_hat = bits_hat.T.copy()  # [Nt, 2*rb]
    bits_hat[bits_hat == -1] = 0  # Ising {-1,1} -> {0,1}

    # QuAMax -> intermediate mapping
    index = np.nonzero(bits_hat[:, rb - 1] == 1)[0]
    bits_hat[index, rb:] = 1 - bits_hat[index, rb:]
    output_bit = bits_hat.copy()
    # differential encoding -> gray
    for i in range(1, nbps):
        output_bit[:, i] = np.logical_xor(bits_hat[:, i], bits_hat[:, i - 1]).astype(np.int32)
    return output_bit.astype(np.int32)

def compute_ber_numpy(solution: np.ndarray, bits_truth: np.ndarray, nbps: int) -> float:
    bits_pred = spins_to_bits_numpy(solution, nbps)  # [Nt, nbps]
    bits_constellation = 1 - np.concatenate([bits_truth[..., 0::2], bits_truth[..., 1::2]], axis=-1)
    # ensure same shapes
    return float(np.mean(bits_constellation != bits_pred))

# ----------------------------
# dataset generator: try to import sionna; if not available, fallback to random channels
# ----------------------------
try:
    from sionna.utils import QAMSource
    from sionna.channel import FlatFadingChannel
    _HAS_SIONNA = True
except Exception:
    _HAS_SIONNA = False

def generate_single_sample(num_tx_ant: int, num_rx_ant: int, SNR: float, num_bits_per_symbol: int):
    """
    Return (H, bits, y) — H complex64 [Nr, Nt], bits int [Nt, nbps], y complex [Nr]
    If sionna available, uses FlatFadingChannel; otherwise random Gaussian channel + AWGN.
    """

    no = num_tx_ant / 10 ** (SNR / 10)
    channel = FlatFadingChannel(num_tx_ant, num_rx_ant, add_awgn=True, return_channel=False)
    h = channel._gen_chn(1)  # [1, Nr, Nt]
    qam_source = QAMSource(num_bits_per_symbol, return_bits=True)
    x, bits = qam_source([1, num_tx_ant])
    y = channel._app_chn([x, h, no])
    h = h.numpy()
    bits = bits.numpy()
    y = y.numpy()
    h = h.squeeze(0)
    bits = bits.squeeze(0)
    y = y.squeeze(0)
    return h.astype(np.complex64), bits.astype(np.int32), y.astype(np.complex64)

# ----------------------------
# Model: DU_GSB_LSTM
# ----------------------------
class DU_GSB_LSTM(nn.Module):
    def __init__(self, T: int, batch_size: int = 100, lstm_hidden=64):
        super().__init__()
        self.T = T
        self.batch_size = batch_size

        # learnable global params
        self.λ = Parameter(torch.tensor([25.0], dtype=torch.float32), requires_grad=True)
        self.Δ_per_step = Parameter(torch.full((T,), 0.01, dtype=torch.float32))
        self.η_per_step = Parameter(torch.full((T,), 1.0, dtype=torch.float32))

        # LSTM controller
        self.lstm_controller = nn.LSTMCell(input_size=3, hidden_size=lstm_hidden)
        self.param_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, 2),   # outputs a, b
            nn.Softplus()
        )

        # Learnable init states
        self.h0 = nn.Parameter(torch.randn(lstm_hidden))
        self.c0 = nn.Parameter(torch.randn(lstm_hidden))

        # normalization buffers
        self.register_buffer('feat_mean', torch.zeros(3))
        self.register_buffer('feat_std', torch.ones(3))
        self.initialized_norm = False

        # prob table
        self.p = 1 - torch.exp(-5 * torch.linspace(0, 1, T))
        i_tensor = torch.arange(self.T, dtype=torch.float32)
        t = i_tensor / float(self.T)
        self.register_buffer('prob_table', 0.1 + 0.45 * (1 + torch.cos(math.pi * t)))

    def get_J_h(self, H: Tensor, y: Tensor, nbps: int) -> Tuple[Tensor, Tensor]:
        # note: self.λ is a tensor shape [1], keep as input
        return to_ising_ext(H, y, nbps, self.λ.to(H.device))

    def forward(self, H: Tensor, y_rx: Tensor, nbps: int, reset_lstm: bool = True) -> Tensor:
        """
        H: complex [Nr, Nt]
        y_rx: complex [Nr]
        returns: spins [B, dim]
        """
        J, h = self.get_J_h(H, y_rx, nbps)
        B = self.batch_size
        N = J.shape[0]  # dim

        c_0 = 0.5 * math.sqrt(max(1, N - 1)) / (torch.linalg.norm(J, ord='fro') + 1e-12)

        # Initialize particles
        x = 0.02 * (torch.rand(N, B, device=H.device) - 0.5)
        y_vel = 0.02 * (torch.rand(N, B, device=H.device) - 0.5)

        energy = calc_energy(x, J, h)
        min_idx = energy.argmin()
        gbest = x[:, min_idx].clone().unsqueeze(1)
        gbest_energy = energy[min_idx].item()
        velocity_memory = torch.zeros_like(y_vel)

        # init lstm states
        if reset_lstm:
            hidden = self.h0.unsqueeze(0).expand(B, -1).clone().to(H.device)
            cell = self.c0.unsqueeze(0).expand(B, -1).clone().to(H.device)
        else:
            # if not reset, still create fresh states (safer)
            hidden = self.h0.unsqueeze(0).expand(B, -1).clone().to(H.device)
            cell = self.c0.unsqueeze(0).expand(B, -1).clone().to(H.device)
            
        for i in range(self.T):
            # update gbest
            current_energy = calc_energy(x, J, h)
            best_idx = current_energy.argmin()
            if current_energy[best_idx] < gbest_energy:
                gbest = x[:, best_idx].clone().unsqueeze(1)
                gbest_energy = float(current_energy[best_idx].item())

            # build features: [B, 3]
            feat1 = current_energy.unsqueeze(0)                     # [1, B]
            feat2 = torch.norm(gbest - x, p=2, dim=0, keepdim=True) # [1, B]
            feat3 = x.std(dim=0, keepdim=True)                      # [1, B]
            features = torch.cat([feat1, feat2, feat3], dim=0).T    # [B, 3]
            
            # lazy normalization
            if not self.initialized_norm:
                with torch.no_grad():
                    feat = features.detach()
                    self.feat_mean.copy_(feat.mean(dim=0))
                    self.feat_std.copy_(feat.std(dim=0).clamp(min=1e-6))
                self.initialized_norm = True

            norm_features = (features - self.feat_mean) / self.feat_std

            # LSTM step
            hidden, cell = self.lstm_controller(norm_features, (hidden, cell))
            ab_params = self.param_head(hidden)  # [B, 2]

            Δ_i = self.Δ_per_step[i].view(1, 1).to(H.device).expand(1, B)
            η_i = self.η_per_step[i].view(1, 1).to(H.device).expand(1, B)

            a_i = (ab_params[:, 0:1].T - 1.0)  # [1, B]
            b_i = (ab_params[:, 1:2].T - 1.0)

            step = i / float(self.T)
            gbest_diff = gbest - x  # [N, B]
            gbest_norm = torch.norm(gbest_diff, dim=0, keepdim=True)
            decay_factor = (1 / (1 + gbest_norm)) * (1 - step)
            info_share_base = decay_factor * gbest_diff

            cos_sim = torch.cosine_similarity(info_share_base, y_vel, dim=0)  # [B]
            soft_mask = torch.tanh(1000 * (cos_sim + b_i.squeeze(0)))  # [B]
            info_share_term = soft_mask.unsqueeze(0) * info_share_base  # [N, B]

            alpha = 0.9 - 0.4 * step
            velocity_memory = alpha * velocity_memory + (1 - alpha) * info_share_term

            # dynamics
            if h is None:
                y_vel = y_vel + (-(1 - self.prob_table[i]) * x + η_i * c_0 * (J @ x) + velocity_memory) * Δ_i
            else:
                y_vel = y_vel + (-(1 - self.prob_table[i]) * x + η_i * c_0 * (J @ x + h) + velocity_memory) * Δ_i

            x = x + Δ_i * y_vel

            # perturbation
            rand_mask = torch.rand_like(x) < self.prob_table[i]
            x = x - rand_mask * a_i

            # activation
            x = φ_s(x)
            y_vel = y_vel * (1 - ψ_s(x))

        return x.T  # [B, dim]

# ----------------------------
# ber_loss for training
# ----------------------------
def ber_loss(spins: Tensor, bits: Tensor, loss_fn: str = 'mse') -> Tensor:
    """
    spins: [dim] or [dim,] in Ising values in [-1,1] (continuous)
    bits: [Nt, nbps] in {0,1} float tensor
    """
    bits_constellation = 1 - torch.cat([bits[..., 0::2], bits[..., 1::2]], dim=-1)
    nbps = bits_constellation.shape[1]
    rb = nbps // 2

    # spins may be [dim] or [B, dim]
    if spins.dim() == 1:
        spins_vec = spins
    else:
        # if [B, dim] pick first candidate
        spins_vec = spins[0]

    spins_reshaped = torch.reshape(spins_vec, (rb, 2, -1))  # [rb, 2, Nt]
    spins_reshaped = torch.permute(spins_reshaped, (2, 1, 0))  # [Nt, 2, rb]
    spins_reshaped = torch.reshape(spins_reshaped, (-1, 2 * rb))  # [Nt, 2*rb]
    bits_hat = (spins_reshaped + 1) / 2

    bits_final = bits_hat.clone()
    index = torch.nonzero(bits_hat[:, rb - 1] > 0.5)[:, -1]
    bits_hat[index, rb:] = 1 - bits_hat[index, rb:]
    for i in range(1, nbps):
        x = bits_hat[:, i] + bits_hat[:, i - 1]
        x_dual = 2 - x
        bits_final[:, i] = torch.where(x <= x_dual, x, x_dual)

    if loss_fn in ['l2', 'mse']:
        return F.mse_loss(bits_final, bits_constellation)
    elif loss_fn in ['l1', 'mae']:
        return F.l1_loss(bits_final, bits_constellation)
    elif loss_fn == 'bce':
        pseudo_logits = bits_final * 2 - 1
        return F.binary_cross_entropy_with_logits(pseudo_logits, bits_constellation)
    else:
        return F.mse_loss(bits_final, bits_constellation)

# ----------------------------
# make_random_transmit wrapper
# ----------------------------
def make_random_transmit(bits_shape: torch.Size, H: Tensor, nbps: int, SNR: int) -> Tuple[Tensor, Tensor]:
    """
    Given bits_shape (Nt, nbps), produce random bits and y using existing H.
    If sionna available, use modulate; otherwise use generate_single_sample fallback.
    """
    Nt = bits_shape[0]
    # fallback: generate new random bits & y using generate_single_sample
    H_np = H.cpu().numpy() if isinstance(H, torch.Tensor) else H
    h_np, bits_np, y_np = generate_single_sample(H_np.shape[1], H_np.shape[0], SNR, nbps)
    bits_t = torch.from_numpy(bits_np).to(device, torch.float32)
    y_t = torch.from_numpy(y_np).to(device, torch.complex64)
    return bits_t, y_t

def train(args):
    """
    Train DU-GSB-LSTM model.
    
    Training strategy:
    1. Randomly sample channel configurations (SNR, antennas, modulation)
    2. Forward pass through model
    3. Compute BER-based loss
    4. Backpropagate through unfolded iterations
    5. Update model parameters
    
    Note: Uses gradient accumulation for stable training.
    """
    print('Device:', device)
    print('Hyperparameters:', vars(args))
    
    # Experiment naming
    exp_name = f'train_{args.M.replace("_", "-")}_T={args.n_iter}_lr={args.lr}_LSTM_origin'
    
    # Dataset configuration space
    SNR_values = [10, 15, 20, 25, 30]
    antenna_configurations = [(16, 16)]
    num_bits_per_symbol_values = [4]  # 16QAM, 64QAM
    
    # Initialize model and optimizer
    model: DU_GSB_LSTM = globals()[args.M](args.n_iter, args.batch_size).to(device)
    optim = Adam(model.parameters(), lr=args.lr)
    
    # Resume training if checkpoint provided
    init_step = 0
    losses = []
    if args.load:
        print(f'>> Resume from {args.load}')
        ckpt = torch.load(args.load, map_location='cpu')
        init_step = ckpt.get('steps', 0)
        losses.extend(ckpt.get('losses', []))
        model.load_state_dict(ckpt['model'], strict=False)
        try:
            optim.load_state_dict(ckpt['optim'])
        except Exception:
            print('Warning: Optimizer state mismatch; using fresh optimizer.')
    
    # Training loop
    loss_wv = []
    steps = init_step
    model.train()
    step_times = []
    try:
        pbar = tqdm(total=args.steps - init_step)
        while steps < init_step + args.steps:
            # Randomly sample channel configuration
            SNR = random.choice(SNR_values)
            tx, rx = random.choice(antenna_configurations)
            nbps = random.choice(num_bits_per_symbol_values)
            
            # Generate sample
            H_np, bits_np, y_np = generate_single_sample(tx, rx, SNR, nbps)
            H = torch.from_numpy(H_np).to(device, torch.complex64)
            y = torch.from_numpy(y_np).to(device, torch.complex64)
            bits = torch.from_numpy(bits_np).to(device, torch.float32)
            
            # Forward pass
            spins = model(H, y, nbps)  # [B, dim]
            
            # Compute loss (average over particles)
            loss_each = torch.stack([ber_loss(sp, bits, args.loss_fn) for sp in spins])
            loss_val = getattr(loss_each, args.agg_fn)()  
            
            # Gradient accumulation
            (loss_val / args.grad_acc).backward()
            
            # Optimizer step
            if steps % args.grad_acc == 0:
                optim.step()
                optim.zero_grad()

            # Logging
            loss_wv.append(loss_val.item())
            steps += 1
            pbar.update(1)
            
            if steps % args.log_every == 0:
                mean_loss = float(np.mean(loss_wv[-args.log_every:]))
                print(f'>> Step {steps} Mean Loss: {mean_loss:.6f}')
                losses.append(mean_loss)
    
    except KeyboardInterrupt:
        print('Training interrupted by user.')
    
    # Save checkpoint
    ckpt = {
        'steps': steps,
        'losses': losses,
        'model': model.state_dict(),
        'optim': optim.state_dict(),
    }
    torch.save(ckpt, LOG_PATH / f'{exp_name}.pth')
    print('Saved checkpoint to', LOG_PATH / f'{exp_name}.pth')
    
    # Plot training loss
    if losses:
        plt.plot(losses)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.tight_layout()
        plt.savefig(LOG_PATH / f'{exp_name}.png', dpi=300)
        print('Saved plot to', LOG_PATH / f'{exp_name}.png')

@torch.no_grad()
def evaluate(args):
    """
    Evaluate trained DU_GSB_LSTM model over multiple SNRs.
    Input:
        --ckpt        path
        --snr         one or more SNR values (e.g., 10 15 20)
        --tx          transmit antennas
        --rx          receive antennas
        --nbps        bits/symbol
        --num_samples number of Monte Carlo trials per SNR
    Output:
        Print BER vs SNR table
    """
    print(">> Evaluation mode (multi-SNR)")
    assert args.ckpt is not None, "请指定 --ckpt 模型路径"
    # Load model once
    ckpt = torch.load(args.ckpt, map_location=device)
    T = ckpt['model']['Δ_per_step'].shape[0]

    model = DU_GSB_LSTM(T, args.batch_size).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    print(f"Model loaded from: {args.ckpt}")
    print(f"Config: Tx={args.eval_tx}, Rx={args.eval_rx}, nbps={args.eval_nbps}, num_samples={args.num_samples}")
    print(f"Evaluating SNRs: {args.eval_snr}")

    results = []
    
    for snr in args.eval_snr:
        print(f"\n--- Evaluating SNR = {snr} dB ---")

        # 构造数据集路径
        mod_map = {2: "QPSK", 4: "QAM16", 6: "QAM64", 8: "QAM256"}
        mod = mod_map.get(args.eval_nbps, f"QAM{2**args.eval_nbps}")
        data_path = f"/home/huangzujia/jugarh/SB/LSTM_DU/dataset/MIMO_{args.eval_tx}x{args.eval_rx}/{mod}/SNR_{snr}/data.npz"
        if not os.path.exists(data_path):
            print(f"Warning: dataset not found: {data_path}")
            continue

        # 加载数据
        data = np.load(data_path)
        H_all = torch.from_numpy(data["H"]).to(device, torch.complex64)
        y_all = torch.from_numpy(data["y"]).to(device, torch.complex64)
        bits_all = data["bits"]  

        total_ber = []
        total_bit_errors = 0
        total_bits = 0

        total_instance_bers = [] 
        total_instance_stds = [] 

        for i in tqdm(range(10000)):
            H = H_all[i]  
            y = y_all[i]
            bits_truth = bits_all[i].copy()
            
            spins = model(H, y, args.eval_nbps, reset_lstm=True)
            spins_sign = torch.sign(spins).cpu().numpy() # [100, N]

            instance_particle_bers = []
            for j in range(args.batch_size):
                p_ber = compute_ber_numpy(spins_sign[j], bits_truth, args.eval_nbps)
                instance_particle_bers.append(p_ber)
            
            instance_particle_bers = np.array(instance_particle_bers)

            instance_mean = np.mean(instance_particle_bers)
            instance_std = np.std(instance_particle_bers)

            total_instance_bers.append(instance_mean)
            total_instance_stds.append(instance_std)

        avg_ber = np.mean(total_instance_bers)
        avg_std = np.mean(total_instance_stds) 
        
        results.append((snr, avg_ber, avg_std))
        print(f"  → Average BER at SNR={snr} dB: {avg_ber:.6e}")
        print(f"  → Average Intra-instance STD: {avg_std:.6e}")

    print("\n" + "="*50)
    print("BER vs SNR Summary")
    print("="*50)
    print(f"{'SNR (dB)':<10} {'BER':<15}")
    print("-"*25)
    for snr, ber, std in results:
        print(f"{snr:<10} {ber:.6e} ± {std:.6e}")
    print("="*50)

    return results


# ----------------------------
# MAIN
# ----------------------------
def build_parser():
    p = ArgumentParser()
    p.add_argument('--mode', type=str, choices=['train', 'eval'], default='eval',
              help="选择运行模式：train 或 eval")
    p.add_argument('--ckpt', type=str, default="/home/huangzujia/jugarh/SB/LSTM_DU/pth/train_DU-GSB-LSTM_Tx32_Rx64_nbps6_snr10-15-20-25-30_T10_bs64_lr0.003_overfit.pth", help="checkpoint path for eval")
    
    # training hparams
    p.add_argument('--M', type=str, default='DU_GSB_LSTM')
    p.add_argument('--n_iter', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=3000)
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--loss_fn', type=str, default='mse')
    p.add_argument('--grad_acc', type=int, default=1)
    p.add_argument('--agg_fn', default='mean', choices=['mean', 'max'])
    p.add_argument('--lr', type=float, default=5e-3)
    p.add_argument('--overfit', type=bool, default=True)
    p.add_argument('--load', type=str, default=None)
    p.add_argument('--log_every', type=int, default=100)
    
    # train args
    p.add_argument('--snr', type=int, nargs='+', default=[30])
    p.add_argument('--tx', type=int, nargs='+', default=[8])
    p.add_argument('--rx', type=int, nargs='+', default=[8])
    p.add_argument('--nbps', type=int, nargs='+', default=[6])

    # eval args
    p.add_argument('--eval_snr', type=int, nargs='+', default=[10,15,20,25,30])
    p.add_argument('--eval_tx', type=int, default=4)
    p.add_argument('--eval_rx', type=int, default=4)
    p.add_argument('--eval_nbps', type=int, default=4)
    p.add_argument('--num_samples', type=int, default=2000)
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        train(args)

    elif args.mode == "eval":
        evaluate(args)

    else:
        print("请选择 --mode train 或 --mode eval")



if __name__ == '__main__':
    main()
