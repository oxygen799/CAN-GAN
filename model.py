# 这一版本写的是盲信道识别，接下来我想尝试一下如果直接知道衰落的参数能不能有好的效果

import torch
import torch.nn as nn
import parameters
import numpy as np

config = parameters.config


# 干扰生成网络的输入应该是高斯噪声
class Generator(nn.Module):
    def __init__(self, message_len=config['message_len']):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(message_len, 2 * message_len),  # [4*12]----->[4*24]
            nn.LeakyReLU(),

            nn.Conv1d(3, 8, 3, 1, padding=1),  # [8*24]
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),

            nn.Conv1d(8, 8, 3, 2, padding=1),  # [8*12]
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),

            nn.Conv1d(8, 16, 1, 1),  # [16*12]
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.Linear(message_len, message_len),  # [16*12]
            nn.LeakyReLU(),

            nn.Conv1d(16, 8, 3, 1, padding=1),  # [8*12]
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),

            nn.Conv1d(8, 2, 3, 1, padding=1),  # [2*12]

            nn.Tanh()
        )

    def forward(self, x, label):
        label_tensor = torch.zeros([x.size(0), 1, config['message_len']]).to(config['device'])
        x = torch.cat([x, label_tensor], dim=1)
        location = torch.where(label == 1)
        location = location[0]
        x[location, 2, :] = 1
        x = self.layers(x)
        return x


class Receiver(nn.Module):
    def __init__(self, message_len=config['message_len']):
        super(Receiver, self).__init__()

        # qpsk解调网络
        self.demoLayers = nn.Sequential(
            nn.Linear(message_len, message_len),  # [2*12]
            nn.LeakyReLU(),

            nn.Conv1d(2, 16, 1, 1),  # [16*12]
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.Conv1d(16, 32, 1, 1),  # [32*12]
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            # 这里的4是因为我使用的QPSK调制,想要转换成为概率问题
            nn.Conv1d(32, 4, 1, 1),  # [4*12]
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),

            nn.Linear(message_len, message_len),  # [4*12]
            nn.Softmax(dim=1)
        )

        # 衰落抵消网络
        self.parameter_est = nn.Sequential(

            nn.Linear(message_len, message_len*2),  # [3*24]
            nn.ReLU(),

            nn.Conv1d(3, 2, 3, 1, 1),  # [2*24]
            nn.ReLU(),

            nn.Linear(message_len * 2, message_len),  # [2*12]
            nn.ReLU(),

            nn.Linear(message_len, 1),  # [2*1]
            nn.Tanh()

        )

    def forward(self, x):
        if config['channel_type'] == 'rayleigh':

            y = torch.cat(
                [x, torch.ones(x.size(0), 1, config['message_len']).to(config['device'])],
                dim=1)
            h = self.parameter_est(y)
            x = transform(x, h)
            x = self.demoLayers(x)
            return x
        else:
            x = self.demoLayers(x)
            return x


class Detector(nn.Module):
    def __init__(self, message_len=config['message_len']):
        super(Detector, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(message_len, message_len),  # [2*12]
            nn.LeakyReLU(),

            nn.Conv1d(2, 16, 3, 1, 1),  # [16*12]
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            nn.Conv1d(16, 32, 3, 1, 1),  # [32
            # *12]
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Conv1d(32, 1, 3, 1, 1),  # [1*12]
            nn.LeakyReLU(),

            nn.Linear(message_len, 1),  # [1*1]
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.layers(x),
        y = torch.stack(y)
        y = y.view(-1)
        return y


def channel(gen_out, cov_out, channel=config['channel_type'], r=1,
            SNR=config['SNR'], h=None):
    if channel == 'awgn':
        return awgn(gen_out, cov_out,
                    SNR)
    if channel == 'rayleigh':
        return rayleigh(gen_out, cov_out, h, SNR, r)
    return gen_out, cov_out


def awgn(gen_out, cov_out, SNR):
    if SNR != None:
        SNR = 10.0 ** (SNR / 10.0)
        noise = torch.randn(*cov_out.size(), requires_grad=False) * (config['covert_data_power'] / SNR / 2) ** 0.5
        noise = noise.to(config['device'])
    else:
        noise = torch.zeros(*cov_out.size(), requires_grad=False)
        noise = noise.to(config['device'])
    return gen_out + cov_out + noise


def rayleigh(gen_out, cov_out, h, SNR,r):
    if SNR != None:
        SNR = 10.0 ** (SNR / 10.0)
        noise = torch.randn(*cov_out.size(), requires_grad=False) * (config['covert_data_power'] / SNR / 2) ** 0.5
        noise = noise.to(config['device'])
    else:
        noise = torch.zeros(*cov_out.size(), requires_grad=False)
        noise = noise.to(config['device'])
    # 由于输入的数据是5000*2*12，不方便使用view_as_complex转换成复数形式，我们改成5000*12*2
    cov_out = cov_out.permute(0, 2, 1).contiguous()
    # 转换成复数形式5000*12
    cov_out = torch.view_as_complex(cov_out)
    if h is None:
        fading_batch = torch.randn((cov_out.size()[0], 1), dtype=torch.cfloat).to(config['device'])
        # fading_batch = fading_batch.unsqueeze(-1)
        # fading_batch = fading_batch.repeat([1,config['message_len']])
    else:
        fading_batch = h
    output_signal = cov_out * fading_batch

    return torch.view_as_real(output_signal).permute(0, 2, 1) + gen_out + noise
    # return torch.view_as_real(output_signal).permute(0,2,1)


def transform(x, h):
    x = x.permute(0, 2, 1).contiguous()
    x = torch.view_as_complex(x)
    h = h.squeeze()  # [5000*2]
    h = torch.view_as_complex(h)  # [5000]
    h = h.unsqueeze(-1)  # [5000*1]
    h = h.repeat(1, config['message_len'])  # [5000*12]

    output_signal = x / h

    return torch.view_as_real(output_signal).permute(0, 2, 1)
