import numpy as np
import scipy.io as sio
import pickle
# receive antennas
MR = 16
# transmit antennas (set not larger than MR!)
MT = 16
# modulation type: 'BPSK','QPSK','16QAM','64QAM'
mod = 'QPSK'
# number of Monte-Carlo trials (transmissions)
trials = 1000
# list of SNR [dB] values to be simulated
SNRdB_list = np.arange(12.5, 15, 2.5)
# set up Gray-mapped constellation alphabet (according to IEEE 802.11)
if mod == 'BPSK':
    symbols = np.array([-1, 1])
elif mod == 'QPSK':
    symbols = np.array([ -1-1j,-1+1j, 1-1j,1+1j ])
elif mod == '16QAM':
    symbols = np.array([ -3-3j,-3-1j,-3+3j,-3+1j, \
               -1-3j,-1-1j,-1+3j,-1+1j, \
               +3-3j,+3-1j,+3+3j,+3+1j, \
               +1-3j,+1-1j,+1+3j,+1+1j ])

elif mod == '64QAM':
    symbols = np.array([ -7-7j,-7-5j,-7-1j,-7-3j,-7+7j,-7+5j,-7+1j,-7+3j, \
                -5-7j,-5-5j,-5-1j,-5-3j,-5+7j,-5+5j,-5+1j,-5+3j, \
                -1-7j,-1-5j,-1-1j,-1-3j,-1+7j,-1+5j,-1+1j,-1+3j, \
                -3-7j,-3-5j,-3-1j,-3-3j,-3+7j,-3+5j,-3+1j,-3+3j, \
                +7-7j,+7-5j,+7-1j,+7-3j,+7+7j,+7+5j,+7+1j,+7+3j, \
                +5-7j,+5-5j,+5-1j,+5-3j,+5+7j,+5+5j,+5+1j,+5+3j, \
                +1-7j,+1-5j,+1-1j,+1-3j,+1+7j,+1+5j,+1+1j,+1+3j, \
                +3-7j,+3-5j,+3-1j,+3-3j,+3+7j,+3+5j,+3+1j,+3+3j ])
elif mod == '256QAM':
#     a = np.repeat(np.arange(-15, 17, 2), 16).reshape(16, 16)
    tmp = np.array([5, 7, 3, 1, 11, 9, 13, 15])
    a = np.repeat(np.concatenate([-1 * tmp[::-1], tmp[::-1]]), 16).reshape(16, 16)
    b = a.T.copy()
    a = a + b*1j
    symbols = a.copy().reshape(-1)
    
elif mod == '1024QAM':
    tmp = np.array([11,9,13,15,5,7,3,1,21,23,19,17,27,25,29,31])
    a = np.repeat(np.concatenate([-1 * tmp[::-1], tmp[::-1]]), 32).reshape(32, 32)
    b = a.T.copy()
    a = a + b*1j
    symbols = a.copy().reshape(-1)
# precompute bit labels
# number of bits per symbol
num_bits_per_sym = int(np.log2(len(symbols)))
#bit_label_file = sio.loadmat('')
bit = np.load('/home/huangzujia/jugarh/SB/LSTM_DU/dataset/%s_%dx%d_hsnr_bit_label.npy' % (mod, MT,MR))
#bit_label = bit_label_file['bits']
# bit_label = np.array([list(bin(item)[2:].zfill(num_bits_per_sym)) 
#                       for item in np.arange(len(symbols))]).astype(np.int)
bit_base = np.array([list(bin(item)[2:].zfill(num_bits_per_sym)) 
                      for item in np.arange(len(symbols))]).astype(int)
# extract average symbol energy
Es = (np.abs(symbols) ** 2).mean()


def ML(H, y):
    # initialization  
    Radius = np.infty
    # path
    PA = np.zeros((MT), dtype=int)
    # stack
    ST = np.zeros((MT, len(symbols)))
    
    # preprocessing
    [Q, R] = np.linalg.qr(H)
    y_hat = Q.conj().T.dot(y)

    # add root node to stack
    level = MT-1
    ST[level,:] = np.abs(y_hat[level]-R[level, level] * symbols) ** 2
    
        # begin sphere decoder
    while (level<MT):
        # find smallest PED in boundary
        idx = np.argmin(ST[level, :])
        minPED = ST[level, idx]

        # only proceed if list is not empty
        if minPED < np.infty:
            # mark child as tested
            ST[level, idx] = np.infty
            # new best path
            new_path = np.concatenate([np.array([idx]), PA[level+1:]])

            # search child
            if minPED < Radius:
                # valid candidate found
                if level > 0:
                    # expand this best node
                    PA[level:] = new_path.copy()
                    # downstep
                    level -= 1
                    DF = R[level, level+1:].dot(symbols[PA[level+1:]])
                    ST[level, :] = minPED + abs(y_hat[level] - R[level, level] * symbols - DF) ** 2
                else:
                    idxML = new_path.astype(int).copy()
                    Radius = minPED
                    
        else:
            level += 1
    
    return symbols[idxML], idxML


ser_list = []
ber_list = []
mse_list = []
#time_list = []
data = pickle.load(open('/home/huangzujia/jugarh/SB/LSTM_DU/dataset/%s_%dx%d_hsnr.pkl'%(mod, MT,MR), 'rb'))
for t in range(trials):
    bit_label = bit[:,:,t]
    for k in range(len(SNRdB_list)):
        #data = sio.loadmat('input32/input_t%d_k%d.mat' % (t+1, k+1))
        index = t * len(SNRdB_list) + k
        Hest = data['Hest'][index]
        y = data['y'][index]
        idx_target =np.array(data['idx'][index])
        s = symbols[idx_target]
        #N0 = Es * np.linalg.norm(Hest, 'fro')** 2 * 10**(-SNRdB_list[k]/10)/MR
        shat, idx = ML(Hest, y)
        target_idxhat = np.argmin(np.abs((shat[:, np.newaxis].dot(np.ones((1, len(symbols))))
         - np.ones((MT, 1)).dot(symbols[np.newaxis,:])) ** 2), 1)

        target_bithat = bit_base[target_idxhat,:]
        # compute error metrics
        target_error = idx_target != target_idxhat
        # symbol error rate
        target_ser = np.sum(target_error)/MT
        # tmp = target_bithat.copy().reshape(-1)
        # tmp[tmp == 0] = -1
        # target_energy = sb.calc_energy(tmp)
        ser = np.sum(target_error)/MT
        #bit error rate
        ber = np.sum(bit_label != target_bithat) / (MT * num_bits_per_sym)
        #mean-squared error
        mse = np.linalg.norm(shat - s) ** 2
        ser_list.append(ser)
        ber_list.append(ber)
        mse_list.append(mse)

        #print('ber: ', ber)
        #print('ser: ', ser)
    print('t: ', t)
    #break

# np.save('1024qam_8x8_hsnr_ml_ser', np.array(ser_list))
np.save('QPSK_16x16_hsnr_ml_ber', np.array(ber_list))
#np.save('qpsk_32x32_ml_mse', np.array(mse_list))


# 加载保存的结果
# ser_results = np.load('1024qam_8x8_hsnr_ml_ser.npy')
ber_results = np.load('QPSK_16x16_hsnr_ml_ber.npy')

# 假设你的 SNR 列表长度是 5 (10, 15, 20, 25, 30)
num_snr = 1 
# 将一维列表转换为 (Trials, SNR) 的矩阵
ber_matrix = ber_results.reshape(-1, num_snr)

# 计算每个 SNR 点的平均 BER
mean_ber = np.mean(ber_matrix, axis=0)

# 打印查看
snr_list = [10]
for i in range(num_snr):
    print(f"SNR: {snr_list[i]}dB | Mean BER: {mean_ber[i]:.6f}")