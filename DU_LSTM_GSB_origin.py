import math
import random
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

import matplotlib.pyplot as plt

# Device configuration: use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------
# ACTIVATION FUNCTIONS
# ----------------------------
# Swish activation: x * sigmoid(x)
σ = torch.sigmoid
sw = lambda x: x * σ(x)

# Smooth sign function for Ising spin representation (-1 to 1)
# Λ controls steepness of transition
φ_s = lambda x, Λ=10: (1 / Λ) * (Λ * sw(x + 1) - Λ * sw(x - 1)) - 1

# Squashing function for velocity damping
# B controls when damping activates, A controls steepness
ψ_s = lambda x, A=100, B=1.01: σ(A * (torch.abs(x) - B))

BASE_PATH = Path(__file__).parent
LOG_PATH = BASE_PATH / 'log'; 
LOG_PATH.mkdir(exist_ok=True)

# ----------------------------
# ISING MODEL CONVERSION
# ----------------------------
def to_ising_ext(H: Tensor, y: Tensor, nbps: int, lmbd: Tensor):
    """
    Convert MIMO detection problem to Ising model (J, h) formulation.
    
    Args:
        H: Channel matrix [Nr, Nt] complex64
        y: Received signal [Nr] complex64
        nbps: Bits per symbol (e.g., 4 for 16QAM)
        lmbd: Regularization parameter (Tikhonov regularization)
    
    Returns:
        J: Coupling matrix [rb*N, rb*N] float32
        h: Bias vector [rb*N, 1] float32
    
    Mathematical Formulation:
        The MIMO detection problem is reformulated as:
            min_s (-0.5 * s^T J s - h^T s)
        where s ∈ {-1, 1}^(rb*N) are Ising spins.
        This comes from maximum likelihood detection with QAM constraints.
    """
    M = 2 ** nbps  # Constellation size
    Nr, Nt = H.shape
    N = 2 * Nt  # Real dimensions (I + Q)
    rb = nbps // 2  # Bits per real dimension
    qam_var = 2 * (M - 1) / 3  # QAM constellation variance
    
    # Identity matrix for transformation
    I = torch.eye(N, device=H.device)
    
    # Build transformation matrix T that maps bits to spins
    T = (2 ** (rb - 1 - torch.arange(rb, device=H.device)))[:, None, None] * I[None, ...]
    T = T.reshape(-1, N).T  # [N, rb*N]
    
    # Convert complex channel to real-valued representation
    # [H_real, -H_imag; H_imag, H_real] * [x_real; x_imag]
    H_tilde = torch.vstack([
        torch.hstack([H.real, -H.imag]),
        torch.hstack([H.imag, H.real]),
    ])  # [2*Nr, 2*Nt]
    
    # Convert complex received signal to real
    y_tilde = torch.cat([y.real, y.imag])  # [2*Nr]
    
    # LMMSE-like regularization for numerical stability
    lmbd_val = lmbd if isinstance(lmbd, torch.Tensor) else torch.tensor([lmbd], device=H.device)
    U_λ = torch.linalg.inv(H_tilde @ H_tilde.T + (lmbd_val * torch.eye(2 * Nr, device=H.device))) / lmbd_val
    
    H_tilde_T = H_tilde @ T  # [2*Nr, rb*N]
    
    J = - H_tilde_T.T @ U_λ @ H_tilde_T * (2.0 / qam_var)
    J = J * (1.0 - torch.eye(J.shape[0], device=J.device))  
    
    ones_rbN = torch.ones((J.shape[0], 1), device=H.device)
    ones_N = torch.ones((N, 1), device=H.device)
    
    z = (y_tilde.unsqueeze(1) - H_tilde_T @ ones_rbN + (math.sqrt(M) - 1) * (H_tilde @ ones_N)) / math.sqrt(qam_var)
    h = 2.0 * H_tilde_T.T @ (U_λ @ z)  # [rb*N, 1]
    
    return J.to(torch.float32), h.to(torch.float32)

# ----------------------------
# ENERGY COMPUTATION
# ----------------------------
def calc_energy(x: Tensor, J: Tensor, h: Tensor) -> Tensor:
    """
    Compute Hamiltonian energy for Ising system.
    
    Args:
        x: Continuous spin values [dim, B]
        J: Coupling matrix [dim, dim]
        h: Bias vector [dim, 1] or None
    
    Returns:
        energy: Energy for each sample in batch [B]
    
    Note:
        Energy E = -0.5 * s^T J s - h^T s
        where s = φ_s(x) are discretized spins
    """
    sign = φ_s(x)  # Discretize continuous values to [-1, 1]
    # Avoid zeros (rare case)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    s = sign.to(device)
    
    As = torch.matmul(J, s)  # [dim, B]
    if h is not None:
        energy = -0.5 * torch.sum(s * As, dim=0) - torch.sum(h * s, dim=0)
    else:
        energy = -0.5 * torch.sum(s * As, dim=0)
    return energy

# ----------------------------
# BIT ERROR RATE COMPUTATION
# ----------------------------
def spins_to_bits_numpy(solution: np.ndarray, nbps: int) -> np.ndarray:
    """
    Convert Ising spin solution to binary bits.
    
    Args:
        solution: Ising spins [-1, 1] shape (rb*2*Nt,)
        nbps: Bits per symbol
    
    Returns:
        bits: Binary bits [Nt, nbps] with values {0, 1}
    
    Process:
        1. Reshape spins to separate real/imaginary and bit positions
        2. Apply differential decoding
        3. Convert Gray-coded bits to binary
    """
    sol = solution.copy().astype(np.int32).reshape((-1,))
    rb = nbps // 2
    Nt = sol.shape[0] // (2 * rb)  # Infer number of transmit antennas
    
    # Reshape: [rb, 2, Nt] for bit positions × (real/imag) × antennas
    bits_hat = sol.reshape(rb, 2, Nt)
    bits_hat = np.concatenate([bits_hat[:, 0], bits_hat[:, 1]], axis=0)  # [2*rb, Nt]
    bits_hat = bits_hat.T.copy()  # [Nt, 2*rb]
    bits_hat[bits_hat == -1] = 0  # Convert Ising {-1,1} to {0,1}
    
    # Apply QuAMax mapping
    index = np.nonzero(bits_hat[:, rb - 1] == 1)[0]
    bits_hat[index, rb:] = 1 - bits_hat[index, rb:]
    output_bit = bits_hat.copy()
    
    # Differential encoding to Gray code
    for i in range(1, nbps):
        output_bit[:, i] = np.logical_xor(bits_hat[:, i], bits_hat[:, i - 1]).astype(np.int32)
    return output_bit.astype(np.int32)

def compute_ber_numpy(solution: np.ndarray, bits_truth: np.ndarray, nbps: int) -> float:
    """
    Compute Bit Error Rate (BER) between predicted and true bits.
    
    Args:
        solution: Ising spin solution
        bits_truth: True bits [Nt, nbps]
        nbps: Bits per symbol
    
    Returns:
        BER: Bit error rate (0.0 to 1.0)
    """
    bits_pred = spins_to_bits_numpy(solution, nbps)  # [Nt, nbps]
    # Reorder bits for comparison (constellation-specific ordering)
    bits_constellation = 1 - np.concatenate([bits_truth[..., 0::2], bits_truth[..., 1::2]], axis=-1)
    return float(np.mean(bits_constellation != bits_pred))

# ----------------------------
# DATASET GENERATION
# ----------------------------
# Try to import Sionna for realistic channel simulation
try:
    from sionna.utils import QAMSource
    from sionna.channel import FlatFadingChannel
    _HAS_SIONNA = True
except Exception:
    _HAS_SIONNA = False

def generate_single_sample(num_tx_ant: int, num_rx_ant: int, SNR: int, num_bits_per_symbol: int):
    """
    Generate single MIMO channel sample.
    
    Args:
        num_tx_ant: Number of transmit antennas
        num_rx_ant: Number of receive antennas
        SNR: Signal-to-Noise Ratio in dB
        num_bits_per_symbol: Constellation size (e.g., 4 for 16QAM)
    
    Returns:
        H: Channel matrix [Nr, Nt] complex64
        bits: Transmitted bits [Nt, nbps] int32
        y: Received signal [Nr] complex64
    
    Note:
        Uses Sionna if available, otherwise falls back to random Gaussian.
    """
    # Use Sionna for realistic channel simulation
    no = num_tx_ant / 10 ** (SNR / 10)  # Noise power
    channel = FlatFadingChannel(num_tx_ant, num_rx_ant, add_awgn=True, return_channel=False)
    h = channel._gen_chn(1)  # [1, Nr, Nt]
    
    # Generate QAM symbols and bits
    qam_source = QAMSource(num_bits_per_symbol, return_bits=True)
    x, bits = qam_source([1, num_tx_ant])
    
    # Apply channel
    y = channel._app_chn([x, h, no])
    
    # Convert to numpy and squeeze batch dimension
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
        
        # Learnable global parameters
        self.λ = Parameter(torch.tensor([25.0], dtype=torch.float32), requires_grad=True)  
        self.Δ_per_step = Parameter(torch.full((T,), 0.01, dtype=torch.float32))  
        self.η_per_step = Parameter(torch.full((T,), 1.0, dtype=torch.float32))  
        
        # LSTM controller for adaptive parameter generation
        self.lstm_controller = nn.LSTMCell(input_size=3, hidden_size=lstm_hidden)
        self.param_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, 2),  # Outputs a, b parameters
            nn.Softplus()  # Ensure positive parameters
        )
        
        # Learnable initial states for LSTM
        self.h0 = nn.Parameter(torch.randn(lstm_hidden))
        self.c0 = nn.Parameter(torch.randn(lstm_hidden))
        
        # Normalization statistics for LSTM inputs
        self.register_buffer('feat_mean', torch.zeros(3))
        self.register_buffer('feat_std', torch.ones(3))
        self.initialized_norm = False
        
        # Probability schedule for perturbation
        self.p = 1 - torch.exp(-5 * torch.linspace(0, 1, T))
        i_tensor = torch.arange(self.T, dtype=torch.float32)
        t = i_tensor / float(self.T)
        self.register_buffer('prob_table', 0.1 + 0.45 * (1 + torch.cos(math.pi * t)))

    def get_J_h(self, H: Tensor, y: Tensor, nbps: int) -> Tuple[Tensor, Tensor]:
        """Convert MIMO problem to Ising model using current λ."""
        return to_ising_ext(H, y, nbps, self.λ.to(H.device))

    def forward(self, H: Tensor, y_rx: Tensor, nbps: int, reset_lstm: bool = True) -> Tensor:
        """
        Forward pass through unfolded optimization.
        
        Args:
            H: Channel matrix [Nr, Nt] complex
            y_rx: Received signal [Nr] complex
            nbps: Bits per symbol
            reset_lstm: Whether to reset LSTM states (for eval)
        
        Returns:
            spins: Continuous spin values for all particles [B, dim]
        """
        # Convert to Ising model
        J, h = self.get_J_h(H, y_rx, nbps)
        B = self.batch_size
        N = J.shape[0]  # Dimension (rb*2*Nt)
        
        # Scale factor for gradient step
        c_0 = 0.5 * math.sqrt(max(1, N - 1)) / (torch.linalg.norm(J, ord='fro') + 1e-12)
        
        # Initialize particles (random starting points)
        x = 0.02 * (torch.rand(N, B, device=H.device) - 0.5)  # Positions
        y_vel = 0.02 * (torch.rand(N, B, device=H.device) - 0.5)  # Velocities
        
        # Compute initial energy and find global best
        energy = calc_energy(x, J, h)
        min_idx = energy.argmin()
        gbest = x[:, min_idx].clone().unsqueeze(1)  # Global best position
        gbest_energy = energy[min_idx].item()
        velocity_memory = torch.zeros_like(y_vel)  # Momentum term
        
        # Initialize LSTM states
        if reset_lstm:
            hidden = self.h0.unsqueeze(0).expand(B, -1).clone().to(H.device)
            cell = self.c0.unsqueeze(0).expand(B, -1).clone().to(H.device)
        else:
            # For training, use fresh states each forward pass
            hidden = self.h0.unsqueeze(0).expand(B, -1).clone().to(H.device)
            cell = self.c0.unsqueeze(0).expand(B, -1).clone().to(H.device)
        
        # Iterate through unfolded steps
        for i in range(self.T):
            # Update global best based on current energy
            current_energy = calc_energy(x, J, h)
            best_idx = current_energy.argmin()
            if current_energy[best_idx] < gbest_energy:
                gbest = x[:, best_idx].clone().unsqueeze(1)
                gbest_energy = float(current_energy[best_idx].item())
            
            # Build features for LSTM controller
            feat1 = current_energy.unsqueeze(0)  # Current energy [1, B]
            feat2 = torch.norm(gbest - x, p=2, dim=0, keepdim=True)  # Distance to best [1, B]
            feat3 = x.std(dim=0, keepdim=True)  # Diversity of particles [1, B]
            features = torch.cat([feat1, feat2, feat3], dim=0).T  # [B, 3]
            
            # Lazy normalization (initialize on first batch)
            if not self.initialized_norm:
                with torch.no_grad():
                    self.feat_mean.copy_(features.mean(dim=0))
                    self.feat_std.copy_(features.std(dim=0).clamp(min=1e-6))
                self.initialized_norm = True
            
            norm_features = (features - self.feat_mean) / self.feat_std
            
            # LSTM forward pass to generate adaptive parameters
            hidden, cell = self.lstm_controller(norm_features, (hidden, cell))
            ab_params = self.param_head(hidden)  # [B, 2] -> (a, b)
            
            # Get current step parameters
            Δ_i = self.Δ_per_step[i].view(1, 1).to(H.device).expand(1, B)  # Step size
            η_i = self.η_per_step[i].view(1, 1).to(H.device).expand(1, B)  # Learning rate
            
            # Adjust a, b parameters
            a_i = (ab_params[:, 0:1].T - 1.0)  # Perturbation strength [1, B]
            b_i = (ab_params[:, 1:2].T - 1.0)  # Velocity alignment parameter [1, B]
            
            # Information sharing mechanism
            step = i / float(self.T)
            gbest_diff = gbest - x  # [N, B]
            gbest_norm = torch.norm(gbest_diff, dim=0, keepdim=True)
            decay_factor = (1 / (1 + gbest_norm)) * (1 - step) 
            info_share_base = decay_factor * gbest_diff
            
            # Velocity alignment using cosine similarity
            cos_sim = torch.cosine_similarity(info_share_base, y_vel, dim=0)  # [B]
            soft_mask = torch.tanh(1000 * (cos_sim + b_i.squeeze(0)))  # [B]
            info_share_term = soft_mask.unsqueeze(0) * info_share_base  # [N, B]
            
            # Update velocity memory 
            alpha = 0.9 - 0.4 * step  
            velocity_memory = alpha * velocity_memory + (1 - alpha) * info_share_term
            
            # Dynamics update
            if h is None:
                y_vel = y_vel + (-(1 - self.prob_table[i]) * x + η_i * c_0 * (J @ x) + velocity_memory) * Δ_i
            else:
                y_vel = y_vel + (-(1 - self.prob_table[i]) * x + η_i * c_0 * (J @ x + h) + velocity_memory) * Δ_i
            
            # Position update
            x = x + Δ_i * y_vel
            
            # Stochastic perturbation 
            rand_mask = torch.rand_like(x) < self.prob_table[i]
            x = x - rand_mask * a_i
            
            # Activation and velocity damping
            x = φ_s(x)  # Constrain to [-1, 1]
            y_vel = y_vel * (1 - ψ_s(x))  # Damp velocity near boundaries
        
        return x.T  # [B, dim]

# ----------------------------
# LOSS FUNCTIONS
# ----------------------------
def ber_loss(spins: Tensor, bits: Tensor, loss_fn: str = 'mse') -> Tensor:
    """
    Compute loss between predicted spins and true bits.
    
    Args:
        spins: Continuous spin values [dim] or [B, dim]
        bits: True bits [Nt, nbps] float tensor (0/1)
        loss_fn: Loss function type ('mse', 'l1', 'bce')
    
    Returns:
        loss: Scalar loss value
    """
    # Reorder bits for constellation comparison
    bits_constellation = 1 - torch.cat([bits[..., 0::2], bits[..., 1::2]], dim=-1)
    nbps = bits_constellation.shape[1]
    rb = nbps // 2
    
    # Handle batch dimension
    if spins.dim() == 1:
        spins_vec = spins
    else:
        spins_vec = spins[0]  # Use first particle for loss
    
    # Convert spins to bit probabilities
    spins_reshaped = torch.reshape(spins_vec, (rb, 2, -1))  # [rb, 2, Nt]
    spins_reshaped = torch.permute(spins_reshaped, (2, 1, 0))  # [Nt, 2, rb]
    spins_reshaped = torch.reshape(spins_reshaped, (-1, 2 * rb))  # [Nt, 2*rb]
    bits_hat = (spins_reshaped + 1) / 2  # Convert [-1,1] to [0,1]
    
    # Apply differential decoding
    bits_final = bits_hat.clone()
    index = torch.nonzero(bits_hat[:, rb - 1] > 0.5)[:, -1]
    bits_hat[index, rb:] = 1 - bits_hat[index, rb:]
    for i in range(1, nbps):
        x = bits_hat[:, i] + bits_hat[:, i - 1]
        x_dual = 2 - x
        bits_final[:, i] = torch.where(x <= x_dual, x, x_dual)
    
    # Compute loss
    if loss_fn in ['l2', 'mse']:
        return F.mse_loss(bits_final, bits_constellation)
    elif loss_fn in ['l1', 'mae']:
        return F.l1_loss(bits_final, bits_constellation)
    elif loss_fn == 'bce':
        pseudo_logits = bits_final * 2 - 1  # Convert [0,1] to [-1,1] for BCE
        return F.binary_cross_entropy_with_logits(pseudo_logits, bits_constellation)
    else:
        return F.mse_loss(bits_final, bits_constellation)

# ----------------------------
# DATA GENERATION WRAPPER
# ----------------------------
def make_random_transmit(bits_shape: torch.Size, H: Tensor, nbps: int, SNR: int) -> Tuple[Tensor, Tensor]:
    """
    Generate random transmit bits and received signal for given channel.
    
    Args:
        bits_shape: Expected shape [Nt, nbps]
        H: Channel matrix
        nbps: Bits per symbol
        SNR: Signal-to-Noise Ratio in dB
    
    Returns:
        bits: Transmitted bits [Nt, nbps]
        y: Received signal [Nr]
    """
    # Fallback to generate_single_sample
    H_np = H.cpu().numpy() if isinstance(H, torch.Tensor) else H
    h_np, bits_np, y_np = generate_single_sample(H_np.shape[1], H_np.shape[0], SNR, nbps)
    bits_t = torch.from_numpy(bits_np).to(device, torch.float32)
    y_t = torch.from_numpy(y_np).to(device, torch.complex64)
    return bits_t, y_t

# ----------------------------
# TRAINING LOOP
# ----------------------------
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
    exp_name = f'train_{args.M.replace("_", "-")}_T={args.n_iter}_lr={args.lr}{"_overfit" if args.overfit else ""}_LSTM_origin'
    
    # Dataset configuration space
    SNR_values = [10, 15, 20, 25, 30]
    antenna_configurations = [(8, 12), (16, 24), (32, 48)]  # (Tx, Rx)
    num_bits_per_symbol_values = [4, 6]  # 16QAM, 64QAM
    
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

# ----------------------------
# EVALUATION
# ----------------------------
@torch.no_grad()
def evaluate(args):
    """
    Evaluate trained model on multiple SNR values.
    
    Evaluation metrics:
        BER: Bit Error Rate averaged over Monte Carlo trials
    
    Args:
        args: Command line arguments with evaluation parameters
    """
    print(">> Evaluation mode (multi-SNR)")
    assert args.ckpt is not None, "Please specify --ckpt model path"
    
    # Load model
    ckpt = torch.load(args.ckpt, map_location=device)
    T = ckpt['model']['Δ_per_step'].shape[0]
    batch_size = 100
    model = DU_GSB_LSTM(T, batch_size).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    
    print(f"Model loaded from: {args.ckpt}")
    print(f"Config: Tx={args.tx}, Rx={args.rx}, nbps={args.nbps}, num_samples={args.num_samples}")
    print(f"Evaluating SNRs: {args.snr}")
    
    results = []
    
    # Evaluate each SNR
    for snr in args.snr:
        print(f"\n--- Evaluating SNR = {snr} dB ---")
        total_ber = []
        
        # Monte Carlo trials
        for i in tqdm(range(args.num_samples), desc=f"SNR {snr}"):
            # Generate test sample
            H_np, bits_np, y_np = generate_single_sample(args.tx, args.rx, snr, args.nbps)
            H = torch.from_numpy(H_np).to(device, torch.complex64)
            y = torch.from_numpy(y_np).to(device, torch.complex64)
            bits_truth = bits_np.copy()
            
            # Convert to Ising model (for energy computation)
            J, h = to_ising_ext(H, y, args.nbps, model.λ)
            
            # Model inference
            spins = model(H, y, args.nbps, reset_lstm=True)
            
            # Discretize spins and compute energy
            spins_sign = torch.sign(spins)  # Convert to {-1, 1}
            if h is None:
                energy = -0.5 * torch.sum(spins_sign @ J * spins_sign, dim=1)
            else:
                energy = -0.5 * torch.sum(spins_sign @ J * spins_sign, dim=1) - (h.T @ spins_sign.T).squeeze(0)
            
            # Select best solution (lowest energy)
            best_idx = energy.argmin().item()
            best_spins = spins_sign[best_idx].cpu().numpy()
            
            # Compute BER
            ber = compute_ber_numpy(best_spins, bits_truth, args.nbps)
            total_ber.append(ber)
        
        avg_ber = np.mean(total_ber)
        results.append((snr, avg_ber))
        print(f"  → Average BER at SNR={snr} dB: {avg_ber:.8f}")
    
    # Print results table
    print("\n" + "="*50)
    print("BER vs SNR Summary")
    print("="*50)
    print(f"{'SNR (dB)':<10} {'BER':<15}")
    print("-"*25)
    for snr, ber in results:
        print(f"{snr:<10} {ber:<15.8f}")
    print("="*50)
    
    return results

# ----------------------------
# COMMAND LINE INTERFACE
# ----------------------------
def build_parser():
    """Build command line argument parser."""
    p = ArgumentParser(description="DU-GSB-LSTM: MIMO Detection with Deep Unfolded Ising Model")
    
    # Mode selection
    p.add_argument('--mode', type=str, choices=['train', 'eval'], default='eval',
                   help="Mode: 'train' or 'eval'")
    p.add_argument('--ckpt', type=str, 
                   default="train_DU-GSB-LSTM_T=10_lr=0.005_overfit_LSTM_origin.pth",
                   help="Checkpoint path for evaluation")
    
    # Training hyperparameters
    p.add_argument('--M', type=str, default='DU_GSB_LSTM', help="Model name")
    p.add_argument('--n_iter', type=int, default=10, help="Number of unfolding iterations (T)")
    p.add_argument('--batch_size', type=int, default=100, help="Batch size (number of particles)")
    p.add_argument('--steps', type=int, default=50000, help="Total training steps")
    p.add_argument('--loss_fn', type=str, default='mse', choices=['mse', 'l1', 'bce'],
                   help="Loss function type")
    p.add_argument('--grad_acc', type=int, default=1, help="Gradient accumulation steps")
    p.add_argument('--agg_fn', default='mean', choices=['mean', 'max'],
                   help="Aggregation function for multi-particle loss")
    p.add_argument('--lr', type=float, default=5e-3, help="Learning rate")
    p.add_argument('--overfit', type=bool, default=True, help="Overfitting mode")
    p.add_argument('--load', type=str, default=None, help="Checkpoint to load")
    p.add_argument('--log_every', type=int, default=100, help="Logging frequency")
    
    # Evaluation parameters
    p.add_argument('--snr', type=int, nargs='+', default=[30], help="SNR values for evaluation")
    p.add_argument('--tx', type=int, default=32, help="Transmit antennas")
    p.add_argument('--rx', type=int, default=48, help="Receive antennas")
    p.add_argument('--nbps', type=int, default=6, help="Bits per symbol")
    p.add_argument('--num_samples', type=int, default=200, help="Monte Carlo trials per SNR")
    
    return p

# ----------------------------
# MAIN ENTRY POINT
# ----------------------------
def main():
    """Main entry point based on command line arguments."""
    parser = build_parser()
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    else:
        print("Please select --mode train or --mode eval")

if __name__ == '__main__':
    main()