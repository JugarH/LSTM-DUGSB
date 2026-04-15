import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import os
import numpy as np
from numpy.random import *
import sys


# global variables

# GPU = True
File = True

# if GPU == True:
#     cuda = torch.device('cuda')
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else 'cpu')
args = sys.argv


# For a Real number system
N = 16  # length of a transmit signal vector
NM_ratio = 1.0 # M/N ratio
M = int(N * NM_ratio) # length of a receive signal vector
batch_size = 1000  # mini-batch size
num_batch = 1000  # number of mini-batches in a generation
num_layers = 10  # number of layers

snr = 10.0  # SNR per receive antenna [dB]
sigma2 = (N/math.pow(10,snr/10.0))/2.0
sigma_std = math.sqrt(sigma2) # SD for w

adam_lr = 0.025 # learning_rate for Adam
test_itr = 1000  # number of iterator for evaluate

file_name = f"N{N}_M{M}_snr{snr}" # File name for write parameter
file_name2 = f"N{N}_M{M}"

# channel_matrix generator
H_re = torch.normal(0.0, std=math.sqrt(0.5) * torch.ones((int)(M/2), (int)(N/2)))
H_im = torch.normal(0.0, std=math.sqrt(0.5) * torch.ones((int)(M/2), (int)(N/2)))
H = torch.cat((torch.cat((H_re,H_im),0),torch.cat((-1*H_im,H_re),0)),1)
Ht = H.t()
if GPU:
    Ht = Ht.cuda()

# change global_H
def H_change():
    global H_re,H_im,H,Ht

    H_re = torch.normal(0.0, std=math.sqrt(0.5) * torch.ones((int)(M/2), (int)(N/2)))
    H_im = torch.normal(0.0, std=math.sqrt(0.5) * torch.ones((int)(M/2), (int)(N/2)))  # sensing matrix
    H = torch.cat((torch.cat((H_re,H_im),0),torch.cat((-1*H_im,H_re),0)),1)
    if GPU:
        H = H.cuda()
        Ht = H.t()
        Ht = Ht.cuda()

def write_param():
    f = open(file_name,"a")
    f.write('\nParameter:\n')
    f.write('--------------------------------------------\n')
    f.write('(N,M)=:({0},{1})\n'.format(N,M))
    f.write('adam_lr:{0}\n'.format(adam_lr))
    f.write('num_layers:{0}\n'.format(num_layers))
    f.write('batch_size:{0}\n'.format(batch_size))
    f.write('num_batch:{0}\n'.format(num_batch))
    f.write('--------------------------------------------\n')
    f.close()

def write_file(BER,gamma,theta,alpha,last_print):
    f = open(file_name, "a")
    f2 = open(file_name2, "a")
    write_param()
    f.write("---------BER---------\n")
    f.write(str(BER))
    f.write("\n---------gamma---------\n")    
    f.write(str(gamma))
    f.write("\n---------theta---------\n")        
    f.write(str(theta))
    f.write("\n---------alpha---------\n")        
    f.write(str(alpha))
    f.write("\n---------last_print---------\n")        
    f.write(str(last_print))
    f.close()
    f2.write(str(BER)+",")
    f2.close()
    return 0

# detection for NaN
def isnan(x):
    return x != x

# mini-batch generator
def generate_batch():
    return 1.0 - 2.0*torch.bernoulli(0.5* torch.ones(batch_size, N))

# definition of TPG-detector network
class TPG_NET(nn.Module):
    def __init__(self):
        super(TPG_NET, self).__init__()
        self.gamma = nn.Parameter(torch.normal(1.0, 0.1 * torch.ones(num_layers)))
        self.theta = nn.Parameter(torch.normal(1.0, 0.1 * torch.ones(num_layers)))
        self.alpha = nn.Parameter(torch.abs(torch.normal(0.0, 0.01 * torch.ones(1))))


    def shrinkage_function(self, y, tau2):  # shrinkage_function
        return torch.tanh(y/tau2)

    def forward(self, x, s, max_itr):  # TPG-detector network
        alpha_I = self.alpha[0]*torch.eye(M).to(device)
        # if GPU:
        # alpha_I = alpha_I.cuda()
        W = Ht.mm((H.mm(Ht) + alpha_I).inverse()) #LMMSE-like matrix
        Wt= W.t()
        y = x.mm(Ht) + torch.normal(0.0, sigma_std*torch.ones(batch_size, M).to(device))
        if GPU:
            y = y.cuda()
        for i in range(max_itr):
            t = y - s.mm(Ht)
            tau2 = torch.abs(self.theta[i])
            r = s + t.mm(Wt)*self.gamma[i]
            s = self.shrinkage_function(r, tau2)
        return s

def eval(network, num_layers_to_eval): #calculate BER
    s_zero = torch.zeros(batch_size, N)
    if GPU:
        s_zero = s_zero.cuda()
    total_errors = 0.0
    total_bits = 0.0
    ber_list = []
    for i in range(test_itr):
        H_change()  # 每次测试使用不同的信道
        x = generate_batch()
        if GPU:
            x = x.cuda()
        x_hat = network(x, s_zero, num_layers_to_eval)
        if isnan(x_hat).any():
            print("Nan detected in evaluation, skipping...")
            continue
        
        # 计算错误比特数
        errors = torch.sum((torch.sign(x_hat) != x).float())
        total_errors += errors.item()
        total_bits += x.numel()
        
    avg_ber = total_errors / total_bits
    print(f"BER = {avg_ber:.6e} ")

    return avg_ber

def main():
    network = TPG_NET()
    if GPU:
        network = network.cuda()  # generating an instance of TPG-detector
    s_zero = torch.zeros(batch_size, N)
    if GPU:
        s_zero = s_zero.cuda()

    # torch.manual_seed(1)

    start = time.time()

    # 训练网络
    print("开始训练...")
    for t in range(num_layers):
        # 为每一层创建新的优化器
        opt = optim.Adam(network.parameters(), lr=adam_lr)
        
        for i in range(num_batch):
            H_change()  # 更新信道
            x = generate_batch()
            if GPU:
                x = x.cuda()

            opt.zero_grad()
            # 使用当前层数进行前向传播
            x_hat = network(x, s_zero, t+1)

            loss = F.mse_loss(x_hat, x) #squared_loss

            if i % 100 == 0:
                print(f'Layer {t+1}, Batch {i}, loss:{loss.item():.6f}')

            loss.backward()

            grads = [param.grad for param in network.parameters()]

            grads_gamma = grads[0]
            grads_theta = grads[1]
            grads_alpha = grads[2]

            if isnan(grads_gamma).any() and isnan(grads_theta).any() and isnan(grads_alpha).any():  # avoiding NaN in gradients
                print("NaN_grad")
                continue
            opt.step()
        
        print(f'Layer {t+1} 训练完成')

    print("所有层训练完成，开始评估...")
    
    # 训练结束后进行最终评估
    final_BER = eval(network, num_layers)
    print(f"最终BER: {final_BER:.6f}")
    
    # 保存参数
    param_set = [param for param in network.parameters()]
    gamma_set = param_set[0]
    theta_set = param_set[1]
    alpha_set = param_set[2]

    gamma_output = ""
    theta_output = ""
    alpha_output = ""
    for i in gamma_set.data.cpu().numpy():
        gamma_output = gamma_output + str(i) + ","
    for i in theta_set.data.cpu().numpy():
        theta_output = theta_output + str(i) + ","
    for i in alpha_set.data.cpu().numpy():
        alpha_output = alpha_output + str(i) + ","

    gamma_output = gamma_output[:-1]
    gamma_output = "[" + gamma_output + "]"

    theta_output = theta_output[:-1]
    theta_output = "[" + theta_output + "]"

    alpha_output = alpha_output[:-1]
    alpha_output = "[" + alpha_output + "]"

    last_print = [f'Final BER: {final_BER:.6f}']
    last_print_BER = [final_BER]

    # if File == True:
    #     write_file(last_print_BER, gamma_output, theta_output, alpha_output, last_print)



if __name__ == '__main__':
    main()