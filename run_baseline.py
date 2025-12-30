import os
import pickle as pkl
from argparse import ArgumentParser
from typing import *

import numpy as np
from numpy import ndarray
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.executing_eagerly()
from sionna.mapping import Constellation, Mapper
from sionna.mimo import EPDetector, KBestDetector, LinearDetector, MaximumLikelihoodDetector, MMSEPICDetector
from sionna.utils import QAMSource
from sionna.channel import FlatFadingChannel

constellation_cache: Dict[int, Constellation] = {}
mapper_cache: Dict[int, Mapper] = {}
detector_cache: Dict[int, Callable] = {}

# https://stackoverflow.com/questions/15505514/binary-numpy-array-to-list-of-integers
bits_to_number = lambda bits: bits.dot(1 << np.arange(bits.shape[-1] - 1, -1, -1))

mean = lambda x: sum(x) / len(x) if len(x) else 0.0

def get_constellation(nbps:int) -> Constellation:
    if nbps not in constellation_cache:
        constellation_cache[nbps] = Constellation('qam', nbps)
    constellation = constellation_cache[nbps]
    #constellation.show() ; plt.show()
    return constellation

def get_mapper(nbps:int) -> Mapper:
    if nbps not in mapper_cache:
        constellation = get_constellation(nbps)
        mapper_cache[nbps] = Mapper(constellation=constellation)
    mapper = mapper_cache[nbps]
    return mapper

def get_detector(args, nbps:int, Nt:int):
    cfg = (nbps, Nt)
    if cfg not in detector_cache:
        # https://nvlabs.github.io/sionna/examples/OFDM_MIMO_Detection.html?highlight=detector#id1
        kwargs = {
            'constellation': get_constellation(nbps),
            'hard_out': True,   # bin_out
        }
        detector_cls = {
            'linear': lambda: LinearDetector(args.E, 'bit', args.D, **kwargs),
            'kbest':  lambda: KBestDetector('bit', Nt, args.k, **kwargs),
            'ep':     lambda: EPDetector('bit', nbps, l=args.l, hard_out=True),
            'mmse':   lambda: MMSEPICDetector('bit', num_iter=args.num_iter, **kwargs),
            # ml does not run due to resource limit (
            'ml':     lambda: MaximumLikelihoodDetector('bit', args.D, Nt, **kwargs),
        }
        detector_cache[cfg] = detector_cls[args.M]()
    return detector_cache[cfg]


def modulate_and_transmit(bits:ndarray, H:ndarray, nbps:int, SNR:int=None) -> Tuple[ndarray, ndarray]:
    mapper = get_mapper(nbps)
    b = tf.convert_to_tensor(bits, dtype=tf.int32)
    x: ndarray = mapper(b).cpu().numpy()

    noise = 0
    if SNR:
        # SNR(dB) := 10*log10(P_signal/P_noise) ?= Var(signal) / Var(noise)
        sigma = np.var(bits) / SNR
        noise = np.random.normal(scale=sigma**0.5, size=H.shape[0])
    y = H @ x + noise[:, np.newaxis] 
    return x, y


def generate(num_tx_ant: int, num_rx_ant: int, SNR: int, num_bits_per_symbol: int, batch_size: int=1):
    # 计算噪声功率
    no = num_tx_ant / 10 ** (SNR / 10)
    channel = FlatFadingChannel(num_tx_ant, num_rx_ant, add_awgn=True, return_channel=False)

    # 生成信道矩阵 h
    h = channel._gen_chn(1)  # [batch_size, num_rx_ant, num_tx_ant]

    # 生成 QAM 符号和比特
    qam_source = QAMSource(num_bits_per_symbol, return_bits=True)
    x, bits = qam_source([batch_size, num_tx_ant])  # x: [batch_size, num_tx_ant], bits: [batch_size, num_tx_ant, num_bits_per_symbol]

    # 通过信道传输并添加噪声
    y = channel._app_chn([x, h, no])  # [batch_size, num_rx_ant]

    # 转换为 numpy 格式
    h = h.numpy()
    bits = bits.numpy()
    y = y.numpy()

    # 如果 batch_size == 1, 改变维度
    if batch_size == 1:
        h = h.squeeze(0)  # [num_rx_ant, num_tx_ant]
        bits = bits.squeeze(0)  # [num_tx_ant, num_bits_per_symbol]
        y = y.squeeze(0)  # [num_rx_ant, ]

    return h, bits, y

def generate_data(num_tx_ant, num_rx_ant, SNR, num_bits_per_symbol):
    h, bits, y = generate(num_tx_ant, num_rx_ant, SNR, num_bits_per_symbol)

    # 将生成的数据存入字典
    data = {
        'H': h.astype(np.complex64),  
        'y': y.astype(np.complex64),  
        'bits': bits.astype(np.int32),  
        'nbps': num_bits_per_symbol, 
        'SNR': SNR  
    }
    return data

def compute_noise_covariance(H, snr_db):
    real_dtype = H.dtype.real_dtype  # complex64 -> float32; complex128 -> float64
    snr_lin = tf.cast(10 ** (snr_db / 10), real_dtype)
    M = tf.cast(tf.shape(H)[-2], real_dtype)
    signal_power = tf.reduce_sum(tf.abs(H)**2, axis=[-2, -1])
    sigma2 = signal_power / (M * snr_lin)
    I_M = tf.eye(tf.shape(H)[-2], dtype=H.dtype)
    return sigma2[..., None, None] * I_M

def run(args):
    
    SNR_values = [25]
    antenna_configurations = [(32, 32)]
    # antenna_configurations = [(8, 12), (16,24), (32,48)]
    num_bits_per_symbol_values  = [2] 
    num_samples = 1000
    results = {}
    
     # 外层进度条：遍历所有配置
    total_configs = len(antenna_configurations) * len(num_bits_per_symbol_values)
    pbar_configs = tqdm(total=total_configs, desc="Configurations", position=0, leave=True)
    
    for config in antenna_configurations:
        tx_ant, rx_ant = config
        for nbps in num_bits_per_symbol_values:
            dataset = []
            key = f"{tx_ant}x{rx_ant}_2^{nbps}-QAM"
            results[key] = []
            # 中层进度条：SNR
            pbar_snr = tqdm(SNR_values, desc=f"{key} | SNR", position=1, leave=False)
            for snr in pbar_snr:
                total_bit_errors = 0
                total_bits = 0
                ber_list = []
                # 内层进度条：num_samples
                pbar_sample = tqdm(range(num_samples), desc=f"SNR={snr}dB", position=2, leave=False, unit="sample")

                for _ in pbar_sample:
                    data = generate_data(tx_ant, rx_ant, snr, nbps)
                    # dataset.append([data['H'], data['y'], data['bits'], data['nbps'], data['SNR']])
                    H: ndarray = data['H']
                    y: ndarray = data['y']
                    bits: ndarray = data['bits'].astype(np.int32)
                    nbps: int = data['nbps']
                    SNR: int = data['SNR']
                    Nr, Nt = H.shape
                    
                    H = tf.convert_to_tensor(H[np.newaxis, ...])    # [B=1, Nr, Nt]
                    y = tf.convert_to_tensor(y.T)                   # [B=1, Nr]


                    snr_linear = 10 ** (snr / 10.0)
                    sigma2 = 1.0 / snr_linear
                    cov = sigma2 * np.eye(Nr, dtype=np.complex64) * Nt
                    s = tf.convert_to_tensor(cov[np.newaxis, ...])

                    Nr = H.shape[1]
                    if y.shape == (Nr):
                        y = y[np.newaxis, :]
                    if not 'debug':
                        print('H.shape:', H.shape)
                        print('y.shape:', y.shape)
                        print('bits.shape:', bits.shape)
                        print('nbps:', nbps)
                        print('SNR:', SNR)
                        # print('ZF_ber:', ZF_ber)

                    detector = get_detector(args, nbps, Nt)
                    if isinstance(detector, MMSEPICDetector):
                        prior_shape = (1, Nt, nbps)  # [B=1, Nt, num_bits_per_symbol]
                        prior = tf.zeros(prior_shape, dtype=tf.float32)
    
                        inputs = (y, H, prior, s)
                    else:
                        inputs = (y, H, s)
                    bits_decode = detector(inputs)[0].cpu().numpy()  # [Nr, nbps]
                    bit_errors = np.sum(bits != bits_decode)
                    total_bit_errors += bit_errors
                    total_bits += bits.size
                    
                avgber = total_bit_errors / total_bits
                stdber = np.sqrt(avgber * (1 - avgber) / total_bits)

                results[key].append(avgber)
                print(f"\n{avgber:.6e}±{stdber:.6e}")
                
            pbar_configs.update(1)  # 完成一个配置
            pbar_configs.set_postfix_str(f"Done: {key}")
                
    pbar_configs.close()
    # 打印表格
    print("BER Benchmark Results (Lower is Better):")
    header = "Config".ljust(15) + "".join([f"SNR={snr:<8}" for snr in SNR_values])
    print(header)
    print("=" * len(header))

    for key, bers in results.items():
        row = key.ljust(15)
        for ber in bers:
            row += f"{ber:.5e}   " if ber is not None else "N/A       "
        print(row)           
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-M', default='linear', choices=['linear', 'kbest', 'ml', 'ml-prior', 'mmse', 'ep'], help='detector')
    parser.add_argument('-E', default='lmmse', choices=['lmmse', 'mf', 'zf'], help='equalizer for LinearDetector')
    parser.add_argument('-D', default='app', choices=['app', 'maxlog'], help='demapping_method')
    parser.add_argument('-k', default=64, type=int, help='k for KBestDetector')
    parser.add_argument('-l', default=10, type=int, help='l for EPDetector')
    parser.add_argument('--num_iter', default=4, type=int, help='num_iter for MMSEPICDetector')
    args = parser.parse_args()

    run(args)
