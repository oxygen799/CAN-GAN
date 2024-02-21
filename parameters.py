import torch

config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'channel_type': 'awgn',
    'fading_power':1,
    # 注意这里的snr和ebn0都是以DB为单位的
    'ebno': 10,
    'SNR': 16,
    'batch_size': 5000,
    'lr': 0.001,
    'n_epochs': 200,
    'early_stop': 300,
    'message_len': 12,
    'Generator_model_link': '../model_record/Generator_model.ckpt',
    'Detector_model_link': '../model_record/Detector_model.ckpt',
    'Receiver_model_link': '../model_record/Receiver_model.ckpt',
    'IC_coefficience': 0.1,
    'Generator_loss_coefficience': 1,
    'n_critic': 1,
    'test_ratio': 2,
    'covert_len': 10000,
    'covert_ratio': 0.5,
    'optimizer': 'RMSprop',
    'covert_data_power': 1,  # 相当于信道强度
    'Gussian_noise_power': 2,
    'channel_noise_power_D': 0.1,
    'channel_noise_power_R': 0.1,
    'G_output_power': 2,  # 相当于干扰的噪声强度
    #一次训练过程中保存的模型数量
    'model_save_num': 20
}