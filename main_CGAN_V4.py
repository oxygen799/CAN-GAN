# 注意，这个是用于做对比的特殊版本，此版本中不再有基于机器学习的AN，而是使用功率均匀分布的AN



import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import math
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.functional as F
import scipy.stats

matplotlib.rc("font", family='DengXian')
import os
from torch.utils.tensorboard import SummaryWriter
import shutil
from parameters import config
import model

os.makedirs('logs', exist_ok=True)
shutil.rmtree('logs')
os.makedirs('../model_record', exist_ok=True)

# DataGenerator.gen(cov_data_num=0.5)

# -------------- read the signal data from file--------------------------
covert_data = pd.read_csv('../covert_message_train.csv', header=None).dropna()
Gussian_noise = pd.read_csv('../Gussian_noise_train.csv', header=None).dropna()
covert_label = pd.read_csv('../covert_message_label_train.csv',
                           header=None).dropna()  # 读取隐蔽信息解码label，label应该是[4*5000*12的矩阵]


def data_process(covert_data, Gussian_noise, covert_label):
    train_data_lenth = Gussian_noise.shape[0]
    cover_data_lenth = covert_data.shape[0]
    covert_ratio = cover_data_lenth / train_data_lenth

    # -------------导入噪声和隐蔽信号之后进行组合，生成一堆维度为[1,12]的数据--------------------
    covert_data_tensor = torch.from_numpy(covert_data.values.astype(complex))  # [5000*12]
    Gussian_noise_tensor = torch.from_numpy(Gussian_noise.values.astype(complex))  # [10000*12],由于隐蔽发送概率是0.5

    # --------------设置隐蔽信号和高斯输入的功率大小------------------------------------------
    covert_data_tensor = covert_data_tensor * config['covert_data_power'] ** 0.5
    Gussian_noise_tensor = Gussian_noise_tensor * config['Gussian_noise_power'] ** 0.5

    zero_tensor = torch.zeros(train_data_lenth - cover_data_lenth, config['message_len'])  # [5000*12]
    covert_data_tensor_with_zero = torch.cat([covert_data_tensor, zero_tensor], dim=0)  # [10000*12]，为了组合形成混合信号

    # -------------------为数据设置label-------------------------------------------------
    label_tensor_one = torch.ones(cover_data_lenth, 1)
    label_tensor_zero = torch.zeros(train_data_lenth - cover_data_lenth, 1)
    label_tensor = torch.cat([label_tensor_one, label_tensor_zero],
                             dim=0)  # [10000*12] 这个是是否含有隐蔽信息的label，前5000*12位为1，后5000*12位为0

    covert_label = covert_label.values
    covert_label = torch.from_numpy(covert_label).long()
    covert_label = covert_label.unsqueeze(0)
    one_hot = torch.zeros(4, covert_label.size(1), covert_label.size(2)).long()
    covert_label_tensor = one_hot.scatter_(0, covert_label, torch.ones_like(covert_label).long())

    covert_label_tensor = torch.cat(
        [covert_label_tensor, torch.zeros([4, train_data_lenth - cover_data_lenth, config['message_len']])],
        dim=1)
    # covert_label_tensor = torch.zeros([4, cover_data_lenth, config['message_len']])
    # label = torch.tensor([[1, 0, 0, 0],
    #                       [0, 1, 0, 0],
    #                       [0, 0, 1, 0],
    #                       [0, 0, 0, 1]])
    # for i in range(4):
    #     location = torch.where(covert_label == i)
    #     for x in range(len(location[0])):
    #         covert_label_tensor[:, location[0][x], location[1][x]] = label[i, :]  # 变成了[4*5000*12]的矩阵，如果原本数据位是0，就是[1,0,0,0]，以此类推
    #
    # covert_label_tensor = torch.cat(
    #     [covert_label_tensor, torch.zeros([4, train_data_lenth - cover_data_lenth, config['message_len']])],
    #     dim=1)  # 组合成为[4*10000*12]的矩阵，没有隐蔽数据的地方就是[0,0,0,0]
    #
    # # -------------后面这一块一开始写的有点问题，所以我先全注释一下，后面用到的话再说----------------------------------
    # #####################################################################################################################################################
    # #####################################################################################################################################################
    # # train_data_tensor = covert_data_tensor_with_zero + Gussian_noise_tensor
    # # train_data_tensor = torch.cat([train_data_tensor,train_label_tensor],dim=1).to(config['device']) #这就是[10000*(12+1)]的训练数据，后面带有标签。
    # #
    # # # 至此我们数据处理完毕，需要的数据已经成为我们的复数tensor，且带有标签位。
    # #
    # # # -------------------但是复数神经网络我没有太多了解，所以想要转换成实数进行运算。-----------------------------
    # # data_tensor_R = torch.cat([torch.real(train_data_tensor),torch.imag(train_data_tensor)],0) # 实部虚部分离之后组合在一起
    # # # reshape成两个channel的形式，也就是[2,5000,(12+1)],带有label，但是原先的label虚部是0，需要重新赋值
    # # data_tensor_R = data_tensor_R.reshape(2,train_data_lenth,config['message_len']+1)
    # # # 把新生成的2channel数据虚数channel的label也给赋值成1
    # # data_tensor_R[1,0:cover_data_lenth,config['message_len']] = 1
    # # # 此时的data都是2个channel，13列，共10000条
    # #####################################################################################################################################################
    # #####################################################################################################################################################

    Gussian_noise_tensor_R = torch.cat([torch.real(Gussian_noise_tensor), torch.imag(Gussian_noise_tensor)], 0).reshape(
        2, train_data_lenth, config['message_len'])
    covert_data_tensor_with_zero_R = torch.cat(
        [torch.real(covert_data_tensor_with_zero), torch.imag(covert_data_tensor_with_zero)], 0).reshape(2,
                                                                                                         train_data_lenth,
                                                                                                         config[
                                                                                                             'message_len'])
    data_tensor_R = torch.cat([Gussian_noise_tensor_R, covert_data_tensor_with_zero_R], dim=0)

    return data_tensor_R, covert_label_tensor, label_tensor


# -----------------------下面编写复数高斯噪声生成网络------------------------------
# -------------------目前水平还不够，所以先编写实数的网络----------------------------
# -------------------信道模型有两种--------------------------------------


# 进行数据的导入，注意这里导入的数据必须为tensor类型。
class Full_duplex_Dataset(Dataset):
    # 将x和y分别导入为feature和label，但是有可能有些问题里没有label，或者是test数据集中没有label
    def __init__(self, x, y=None, label=None):
        self.x = x
        self.y = y
        self.label = label

    def __len__(self):
        return self.x.size(1)

    def __getitem__(self, idx):
        return self.x[:, idx, :], self.y[:, idx, :], self.label[idx]


train_data_tensor_R, train_covert_label_tensor, train_label_tensor = data_process(covert_data, Gussian_noise,
                                                                                  covert_label)
train_dataset = Full_duplex_Dataset(train_data_tensor_R, train_covert_label_tensor, train_label_tensor)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)


# 接下来开始写训练过程

class Trainer():
    def __init__(self, config):
        self.config = config
        self.Generator = model.Generator().to(config['device'])
        self.Receiver = model.Receiver().to(config['device'])
        self.Detector = model.Detector().to(config['device'])
        self.loss_BCE = nn.BCELoss()
        self.loss_CE = nn.CrossEntropyLoss()
        if config['optimizer'] == 'RMSprop':
            self.opt_Generator = torch.optim.RMSprop(self.Generator.parameters(), lr=self.config['lr'])
            self.opt_Receiver = torch.optim.RMSprop(self.Receiver.parameters(), lr=self.config['lr'])
            self.opt_Detector = torch.optim.RMSprop(self.Detector.parameters(), lr=self.config['lr'])
        elif config['optimizer'] == 'Adam':
            self.opt_Generator = torch.optim.Adam(self.Generator.parameters(), lr=self.config['lr'])
            self.opt_Receiver = torch.optim.Adam(self.Receiver.parameters(), lr=self.config['lr'])
            self.opt_Detector = torch.optim.Adam(self.Detector.parameters(), lr=self.config['lr'])

    def start_train(self, train_loader, device=config['device']):

        Estimate = 0
        n_epochs = self.config['n_epochs']
        model_name_num = 0
        writer = {
            'Detector': SummaryWriter("../logs/Detector"),
            'Receiver': SummaryWriter("../logs/Receiver"),
            'Generator': SummaryWriter("../logs/Generator")
        }
        step = 0

        for epoch in range(n_epochs):

            for batches_done, (data, QPSK_label, label) in enumerate(train_loader):

                data = data.to(device)
                label = label.squeeze().to(device)
                QPSK_label = QPSK_label.to(device)

                Gussian_input = data[:, 0:2, :].float()  # [5000,2,12]
                covert_input = data[:, 2:4, :].float()  # [5000,2,12],包含有0，未知的隐蔽位置
                step += 1

                # 生成网络的输出一定要乘上功率因子，不然容易性能受限
                # if (config['channel_type'] == 'rayleigh'):
                #     Gen_output = self.Generator(Gussian_input, label,Estimate) * math.sqrt(config['G_output_power'])
                # else:
                Gen_output = self.Generator(Gussian_input, label) * math.sqrt(config['G_output_power'])
                paper_unidis_noise=torch.randn((config['batch_size'],1,config['message_len']), dtype=torch.cfloat).to(config['device'])
                paper_unidis=torch.rand((config['batch_size'],1,config['message_len'])).to(config['device'])*config['G_output_power']
                paper_unidis_noise=paper_unidis_noise*paper_unidis
                paper_unidis_noise=paper_unidis_noise.permute(0,2,1)
                paper_unidis_noise=torch.view_as_real(paper_unidis_noise).squeeze()
                paper_unidis_noise=paper_unidis_noise.permute(0,2,1)
                Gen_output=paper_unidis_noise

                covert_output = covert_input
                # ---------------------------------
                # trian Detector
                # ---------------------------------
                self.opt_Detector.zero_grad()
                Detector_input = model. \
                    channel(Gen_output.detach(), covert_output)
                Detector_output = self.Detector(Detector_input)
                loss_Detector = self.loss_BCE(Detector_output, label)
                D_accuracy_label = Detector_output > 0.5
                accuracy_Detector = torch.sum(D_accuracy_label == label) / label.size(0)
                if epoch % config['n_critic'] == 0:
                    # 反向传播计算梯度
                    loss_Detector.backward()
                    # 进行更新
                    self.opt_Detector.step()
                    # 记录训练过程
                    writer['Detector'].add_scalar('Accuracy', accuracy_Detector, step)
                # ---------------------------------
                # trian Receiver
                # ---------------------------------
                self.opt_Receiver.zero_grad()
                covert_location = torch.where(
                    label == 1)  # 2023.6.30改，这里是为了让接收端的输入一定是含有隐蔽信息的信号，因为接收端只需要衡量解调正确性，给出正确的QPSKlabel
                covert_location = covert_location[0]
                Receiver_input = model.channel(
                    Gen_output[covert_location, :, :].detach() * config['IC_coefficience'], covert_output[
                                                                                            covert_location, :,
                                                                                            :])  # ？？？？？？？这里是否要体现自干扰抵消系数？？？？？

                Receiver_output = self.Receiver(Receiver_input)
                loss_Receiver = self.loss_CE(Receiver_output, QPSK_label[covert_location, :, :])
                loss_Receiver.backward()
                self.opt_Receiver.step()
                Receiver_output_label = torch.argmax(Receiver_output, dim=1)
                acc_QPSK_label = torch.argmax(QPSK_label[covert_location, :, :], dim=1)
                accuracy_Receiver = torch.sum(Receiver_output_label.eq(acc_QPSK_label)) / (
                        len(covert_location) * config['message_len'])
                writer['Receiver'].add_scalar('Accuracy', accuracy_Receiver, step)
                # ----------------------------------
                # trian Generator
                # ----------------------------------
                self.opt_Generator.zero_grad()
                Detector_input_withoutdetach = model.channel(Gen_output, covert_output)
                Receiver_input_withoutdetach = model.channel(
                    Gen_output[covert_location, :, :] * config['IC_coefficience'], covert_output[covert_location, :,
                                                                                   :])

                loss_Generator = -self.loss_BCE(self.Detector(Detector_input_withoutdetach), label) + self.loss_CE(
                    self.Receiver(Receiver_input_withoutdetach),
                    QPSK_label[covert_location, :, :])  # 与Receiver的区别就是有没有detach()
                loss_Generator.backward()
                self.opt_Generator.step()
                writer['Generator'].add_scalar('Loss', loss_Generator, step)
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [Detector loss: %f] [Detector acc: %f] [Receiver loss: %f] [Receiver acc: %f] [Generator loss: %f]"
                    % (epoch + 1, n_epochs, batches_done + 1, len(train_loader), loss_Detector.item(),
                       accuracy_Detector.item(),
                       loss_Receiver.item(), accuracy_Receiver.item(), loss_Generator.item())
                )

                # 多次保存模型

            if (epoch + 1) / n_epochs in [(i+1)/config['model_save_num'] for i in range(config['model_save_num'])]:
                torch.save(self.Receiver.state_dict(), f'{config["Receiver_model_link"]}_{model_name_num}')
                torch.save(self.Detector.state_dict(), f'{config["Detector_model_link"]}_{model_name_num}')
                torch.save(self.Generator.state_dict(), f'{config["Generator_model_link"]}_{model_name_num}')
                model_name_num += 1
                print('正在记录训练模型……')


# --------------------------- 测试模块 --------------------------------------------

# 绘图模块，可以绘制隐蔽和非隐蔽的信号分布图
def draw_pic(public_location, covert_location, Generator_pred, Detector_input, Receiver_input,covert_output, plt_num=100):
    Generator_pred_covert = Generator_pred[covert_location, :, :].to('cpu')
    Generator_pred_public = Generator_pred[public_location, :, :].to('cpu')
    Detector_input_covert = Detector_input[covert_location, :, :].to('cpu')
    Detector_input_public = Detector_input[public_location, :, :].to('cpu')
    Receiver_input_covert = Receiver_input[:, :, :].to('cpu')
    covert_output = covert_output[covert_location, :, :].to('cpu')

    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    ## ==============================绘制星座图=========================================================
    plt.figure(1)
    # ============
    plt.subplot(221)
    plt.scatter(Generator_pred_covert[0:plt_num, 0, :].reshape(-1), Generator_pred_covert[0:plt_num, 1, :].reshape(-1),s=2)
    plt.ylabel("Im",fontsize=14)
    plt.xlabel("Re",fontsize=14)
    plt.title('Covert Slot AN',fontname="Times New Roman",fontweight='bold',fontsize='large')

    plt.subplot(222)
    plt.scatter(Generator_pred_public[0:plt_num, 0, :].reshape(-1), Generator_pred_public[0:plt_num, 1, :].reshape(-1),s=2)
    plt.ylabel("Im",fontsize=14)
    plt.xlabel("Re",fontsize=14)
    plt.title('Silent Slot AN',fontname="Times New Roman",fontweight='bold',fontsize='large')

    plt.subplot(223)
    plt.scatter(Detector_input_covert[0:plt_num, 0, :].reshape(-1), Detector_input_covert[0:plt_num, 1, :].reshape(-1),s=2)
    plt.ylabel("Im",fontsize=14)
    plt.xlabel("Re",fontsize=14)
    plt.title('Covert Slot Signal',fontname="Times New Roman",fontweight='bold',fontsize='large')

    plt.subplot(224)
    plt.scatter(Detector_input_public[0:plt_num, 0, :].reshape(-1), Detector_input_public[0:plt_num, 1, :].reshape(-1),s=2)
    plt.ylabel("Im",fontsize=14)
    plt.xlabel("Re",fontsize=14)
    plt.title('Silent Slot Signal',fontname="Times New Roman",fontweight='bold',fontsize='large')

    # plt.subplot(325)
    # plt.scatter(Receiver_input_covert[0:plt_num, 0, :].reshape(-1), Receiver_input_covert[0:plt_num, 1, :].reshape(-1),s=2)
    # plt.title('隐蔽  R的输入')
    #
    # plt.subplot(326)
    # plt.scatter(covert_output[0:plt_num, 0, :].reshape(-1), covert_output[0:plt_num, 1, :].reshape(-1), s=2)
    # plt.title('原始信号')

    ## =================================绘制频率分布直方图====================================

    plt.figure(2)
    # ============
    plt.subplot(2, 2, 1)
    temp = (Generator_pred_covert[:, 0, :] ** 2 + Generator_pred_covert[:, 1, :] ** 2) ** 0.5
    temp = temp.reshape(-1)
    plt.hist(temp, bins=1000, density=True)
    plt.xlabel("Power",fontsize=14)
    plt.ylabel("Density",fontsize=14)
    plt.title('Covert Slot AN Distribution',fontname="Times New Roman",fontweight='bold',fontsize='large')

    plt.subplot(2, 2, 2)
    temp = (Generator_pred_public[:, 0, :] ** 2 + Generator_pred_public[:, 1, :] ** 2) ** 0.5
    temp = temp.reshape(-1)
    plt.hist(temp, bins=1000, density=True)
    plt.xlabel("Power", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title('Silent Slot AN Distribution',fontname="Times New Roman",fontweight='bold',fontsize='large')

    plt.subplot(2, 2, 3)
    temp = (Detector_input_covert[:, 0, :] ** 2 + Detector_input_covert[:, 1, :] ** 2) ** 0.5
    temp = temp.reshape(-1)
    plt.hist(temp, bins=1000, density=True)
    plt.xlabel("Power", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title('Covert Slot Signal Power Distribution',fontname="Times New Roman",fontweight='bold',fontsize='large')

    plt.subplot(2, 2, 4)
    temp = (Detector_input_public[:, 0, :] ** 2 + Detector_input_public[:, 1, :] ** 2) ** 0.5
    temp = temp.reshape(-1)
    plt.hist(temp, bins=1000, density=True)
    plt.xlabel("Power", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title('Silent Slot Signal Power Distribution',fontname="Times New Roman",fontweight='bold',fontsize='large')

    # plt.subplot(3, 2, 5)
    # temp = (Detector_input_covert[:, 0, :] ** 2 + Detector_input_covert[:, 1, :] ** 2) ** 0.5
    # temp = temp.reshape(-1)
    # plt.hist(temp, bins=1000, density=True, cumulative=True)
    # plt.title('隐蔽  D输入分布cdf')
    #
    # plt.subplot(3, 2, 6)
    # temp = (Detector_input_public[:, 0, :] ** 2 + Detector_input_public[:, 1, :] ** 2) ** 0.5
    # temp = temp.reshape(-1)
    # plt.hist(temp, bins=1000, density=True, cumulative=True)
    # plt.title('公开  D输入分布cdf')

    plt.show()


def Compute_Divergence(public_location, covert_location,  Detector_input ,bins=5000):
    Detector_input_covert = Detector_input[covert_location, :, :].to('cpu')
    Detector_input_public = Detector_input[public_location, :, :].to('cpu')

    temp1 = (Detector_input_covert[:, 0, :] ** 2 + Detector_input_covert[:, 1, :] ** 2) ** 0.5
    temp1 = temp1.reshape(-1)

    temp2 = (Detector_input_public[:, 0, :] ** 2 + Detector_input_public[:, 1, :] ** 2) ** 0.5
    temp2 = temp2.reshape(-1)

    hist1,xedges1, yedges1 = np.histogram2d(Detector_input_covert[:, 0, :].reshape(-1).numpy(), Detector_input_covert[:, 1, :].reshape(-1).numpy(), [100, 100])
    hist2,xedges2, yedges2 = np.histogram2d(Detector_input_public[:, 0, :].reshape(-1).numpy(), Detector_input_public[:, 1, :].reshape(-1).numpy(), [100, 100])
    # 转化为概率
    hist1 = hist1 / 10000
    hist2 = hist2 / 10000
    # 展开为一维方便计算
    hist1 = hist1.flatten()
    hist2 = hist2.flatten()

    M = (hist1 + hist2) / 2
    JSD_2D = 0.5 * scipy.stats.entropy(hist1, M) + 0.5 * scipy.stats.entropy(hist2, M)



    # 通过频率计算概率
    n, bin_edges = np.histogram(temp1, bins=bins)
    totalcount = np.sum(n)
    bin_probability_1 = n / totalcount
    n, bin_edges = np.histogram(temp2, bins=bins)
    totalcount = np.sum(n)
    bin_probability_2 = n / totalcount

    # 计算推土机距离，前两个参数表示位置坐标，后两个参数表示分别的土堆高度
    dists = [i for i in range(bins)]
    WD = scipy.stats.wasserstein_distance(dists, dists, bin_probability_1, bin_probability_2)

    # 计算JS散度进行尝试
    M = (bin_probability_1 + bin_probability_2) / 2
    JSD = 0.5 * scipy.stats.entropy(bin_probability_1, M) + 0.5 * scipy.stats.entropy(bin_probability_2, M)

    return WD, JSD , JSD_2D

# 测试模块，可自动导入测试数据并且开始测试，返回准确率和预测结果
def test(draw=True):
    test_Generator_model = model.Generator().to(config['device'])
    test_Generator_model.load_state_dict(torch.load(config['Generator_model_link']))

    test_Detector_model = model.Detector().to(config['device'])
    test_Detector_model.load_state_dict(torch.load(config['Detector_model_link']))

    test_Receiver_model = model.Receiver().to(config['device'])
    test_Receiver_model.load_state_dict(torch.load(config['Receiver_model_link']))

    covert_data = pd.read_csv('../covert_message_test.csv', header=None).dropna()
    Gussian_noise = pd.read_csv('../Gussian_noise_test.csv', header=None).dropna()
    covert_label = pd.read_csv('../covert_message_label_test.csv', header=None).dropna()

    test_data_tensor_R, test_covert_label_tensor, test_label_tensor = data_process(covert_data, Gussian_noise,
                                                                                   covert_label)

    test_dataset = Full_duplex_Dataset(test_data_tensor_R, test_covert_label_tensor, test_label_tensor)
    n_test = int(config['test_ratio'] * config['covert_len'] / config['covert_ratio'])
    test_loader = DataLoader(test_dataset, batch_size=n_test, shuffle=True, drop_last=True)

    with torch.no_grad():
        for data, QPSK_label, label in test_loader:
            test_Generator_model.eval()
            test_Detector_model.eval()
            test_Receiver_model.eval()

            data = data.to(config['device'])
            label = label.squeeze().to(config['device'])
            QPSK_label = QPSK_label.to(config['device'])

            Gussian_input = data[:, 0:2, :].float()
            covert_input = data[:, 2:4, :].float()



            covert_location = torch.where(label == 1)
            covert_location = covert_location[0]
            public_location = torch.where(label != 1)
            public_location = public_location[0]

            # 此信号仅仅用于观测不加人工噪声的原始信号
            covert_output = model.channel(0, covert_input)
            # 生成网络的输出一定要乘上功率因子，不然容易性能受限
            Generator_pred = test_Generator_model(Gussian_input, label) * math.sqrt(config['G_output_power'])
            paper_unidis_noise = torch.randn((config['batch_size']*8, 1, config['message_len']), dtype=torch.cfloat).to(
                config['device'])
            paper_unidis = torch.rand((config['batch_size']*8, 1, config['message_len'])).to(config['device']) * config[
                'G_output_power']
            paper_unidis_noise = paper_unidis_noise * paper_unidis
            paper_unidis_noise = paper_unidis_noise.permute(0, 2, 1)
            paper_unidis_noise = torch.view_as_real(paper_unidis_noise).squeeze()
            paper_unidis_noise = paper_unidis_noise.permute(0, 2, 1)
            Generator_pred = paper_unidis_noise
            Detector_input = model.channel(Generator_pred, covert_input)
            Receiver_input = model.channel(
                Generator_pred[covert_location, :, :] * config['IC_coefficience'] , covert_input[
                                                                                    covert_location, :,
                                                                                    :])
            Detector_pred = test_Detector_model(Detector_input)
            Receiver_pred = test_Receiver_model(Receiver_input)
            if draw == True:
                draw_pic(public_location, covert_location, Generator_pred, Detector_input, Receiver_input,covert_output ,100)

            WD,JSD,JSD_2D = Compute_Divergence(public_location, covert_location, Detector_input)

            Detector_pred[Detector_pred < 0.5] = 0
            Detector_pred[Detector_pred >= 0.5] = 1

            Receiver_output_label = torch.argmax(Receiver_pred, dim=1)
            acc_QPSK_label = torch.argmax(QPSK_label[covert_location, :, :], dim=1)

            Receiver_accuracy = torch.sum(Receiver_output_label.eq(acc_QPSK_label)) / (
                    len(covert_location) * config['message_len'])

            Detector_accuracy = torch.sum(Detector_pred.eq(label)) / len(label)

            print(f'Detector accuracy: {float(Detector_accuracy):.4f} \nReceiver accuracy: {float(Receiver_accuracy):.4f}'
                  f'\nWasserstein Divergence:{float(WD):.4f}\nJS Divergence{float(JSD):.4f}'
                  f'\nWasserstein Divergence 2D: {float(JSD_2D):.4f} \n')
            return WD,JSD,JSD_2D


# # --------------------------- 开始训练 --------------------------------------------
trainer = Trainer(config)
trainer.start_train(train_loader, config['device'])
print('ok')


# 选择模型模块，可以选择一个保存的模型
def choose_model(model_index=4):
    if os.path.exists(config['Generator_model_link']):
        os.remove(config['Generator_model_link'])
    if os.path.exists(config['Detector_model_link']):
        os.remove(config['Detector_model_link'])
    if os.path.exists(config['Receiver_model_link']):
        os.remove(config['Receiver_model_link'])

    shutil.copy(f"{config['Generator_model_link']}_{model_index}", '../model_record/Generator_model.ckpt')
    shutil.copy(f"{config['Detector_model_link']}_{model_index}", '../model_record/Detector_model.ckpt')
    shutil.copy(f"{config['Receiver_model_link']}_{model_index}", '../model_record/Receiver_model.ckpt')

    return None

# 绘制JD散度和推土机距离的代码
# a = []
# b = []
# c = []
#
# for i in range(config['model_save_num']):
#     choose_model(i)
#     WD,JSD,JS_2D = test(draw=False)
#     a.append(WD)
#     b.append(JSD)
#     c.append(JS_2D)
# figure = plt.figure
# plt.subplot(311)
# plt.plot(a)
# plt.subplot(312)
# plt.plot(b)
# plt.subplot(313)
# plt.plot(c)
# plt.show()




# 绘制分布和星座图的代码
choose_model(18)
test(draw=False)


print('ok')
