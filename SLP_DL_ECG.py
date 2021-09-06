import os
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,TensorDataset,random_split
import torch.nn.functional as F
import pytorch_lightning as pl 
import numpy as np
import math
import glob
import os
import pandas as pd
import random
import mne
from scipy import signal
from sklearn import preprocessing

# ['EEG C3-CLE', 'EEG C4-CLE', 'EEG O1-CLE', 'EEG O2-CLE', 'EOG Upper Vertic', 'EOG Lower Vertic', 'EOG Left Horiz', 'EOG Right Horiz', 'EMG Chin',
#  'ECG ECGI', 'ECG ECGII', 'ECG ECGIII', 'Resp Nasal', 'EEG A2-CLE', 'SaO2 SaO2']
    

# class MassDataset(Dataset):
#     def  __init__(
#         self,
#         data_dir=None,
#         eval_ratio=0.2,
#     ):
#         super().__init__()
#         self.data_dir = data_dir
#         self.eval_ratio = eval_ratio
        
#         biosig_files = glob.glob(os.path.join(self.data_dir,'*PSG.edf'))    #### Biosignal files contain Polysomnography signals
#         biosig_files.sort()
# #         random.shuffle(biosig_files)
#         # biosig_files=biosig_files[0:10]   #### USING PART OF THE DATA
#         ## Creating a 80-20 split ###
#         ids = [i for i in range(len(biosig_files))]
#         random.shuffle(ids)
#         train_ids = ids[:int(len(biosig_files) * 0.8)]
#         test_ids = ids[int(len(biosig_files) * 0.8):]
        
#         self.ecg_train2 = torch.Tensor([])
#         self.eeg_trainc3a2 = torch.Tensor([])
#         self.slp_stg_train = torch.Tensor([])
        
#         self.ecg_test2 = torch.Tensor([])
#         self.eeg_testc3a2 = torch.Tensor([])
#         self.slp_stg_test = torch.Tensor([])
        
#         self.n_slpstg=0
        
#         # import pdb;pdb.set_trace()
#         for (index,file) in enumerate(biosig_files):
            
#             ### ANNOTATION PROCESSING
#             annot_filename = (os.path.join(os.path.dirname(biosig_files[0]), file[-18:-8] + ' Base.edf'))
#             base4 = mne.read_annotations(annot_filename)  ##### BASE FILE CONTAIN SLEEP STAGES
#             sleep_stages = [sleep_stage[-1] for sleep_stage in base4.description]
#             slp_coarse_map = {"?" : -2,"W" : 0,"1" : 1,"2" : 1,"3" : 2,"4" : 2,"R" : 3}
#             slp_stg_coarse = list(map(slp_coarse_map.get,sleep_stages))
#             sleep_epoch_len = 20  # in seconds
            
#             ### BIOSIGNAL PROCESSING
#             psg4 = mne.io.read_raw_edf(file)
# #             srate = psg4.info['sfreq']
# #             resp4=psg4['Resp Nasal'][0][0]
#             # ecg1= psg4['ECG ECGI'][0][0]
#             # ecg2= psg4['ECG ECGII'][0][0]
#             # ecg3= psg4['ECG ECGIII'][0][0]
#             # eegc3= psg4['EEG C3-CLE'][0][0]
#             # eegc4= psg4['EEG C4-CLE'][0][0]
#             # eego1= psg4['EEG O1-CLE'][0][0]
#             # eego2= psg4['EEG O2-CLE'][0][0]
#             # eega2= psg4['EEG A2-CLE'][0][0]

#             ecg2= psg4['ECG ECGII'][0][0]    # Inverse of ideal waveform was available from dataset so correcting it by * (-1)
#             eegc3a2= psg4['EEG C3-CLE'][0][0] - psg4['EEG A2-CLE'][0][0]  # Getting C3 channel with A2 reference
#             fs = int(psg4.info['sfreq'])
            
#             ## Resampling 
#             resamp_srate=200
#             num_windows = int(min(len(ecg2) / fs / sleep_epoch_len, len(base4)))
            
#             min_max_scaler = preprocessing.MinMaxScaler()
#             #windowed_ecg2 = [signal.resample(np.transpose(min_max_scaler.fit_transform(np.transpose(ecg2[i * fs * sleep_epoch_len:(i+1) * fs * sleep_epoch_len], (1,0)) (1,0))), resamp_srate*sleep_epoch_len) for i in range(num_windows)] 
#             # import pdb;pdb.set_trace()
#             # windowed_ecg2 = []
#             # windowed_ecg2 = [signal.resample(-1*ecg2[i * fs * sleep_epoch_len:(i+1) * fs * sleep_epoch_len], resamp_srate*sleep_epoch_len) for i in range(num_windows)]# if fs != resamp_srate]
            
#             # windowed_ecg2 = [signal.resample(min_max_scaler.fit_transform(ecg2[i * fs * sleep_epoch_len:(i+1) * fs * sleep_epoch_len].reshape(1,-1)).T.squeeze(1), resamp_srate*sleep_epoch_len) for i in range(num_windows)]# if fs != resamp_srate]
            
#             ## NORMALIZATION AND WINDOWING OF INPUT SIGNALS
#             windowed_ecg2 = [signal.resample(min_max_scaler.fit_transform(ecg2[i * fs * sleep_epoch_len:(i+1) * fs * sleep_epoch_len].reshape(-1,1)).T.squeeze(0), resamp_srate*sleep_epoch_len) for i in range(num_windows)]# if fs != resamp_srate]
#             windowed_eegc3a2 = [signal.resample(min_max_scaler.fit_transform(eegc3a2[i * fs * sleep_epoch_len:(i+1) * fs * sleep_epoch_len].reshape(-1,1)).T.squeeze(0), resamp_srate*sleep_epoch_len) for i in range(num_windows)]# if fs != resamp_srate]
            
#             # plt.plot(windowed_ecg2[120])
#             # plt.show()
#             # for i in tqdm(range(1, num_windows)):
#             #     try:
#             #         windowed_ecg2.append(signal.resample(min_max_scaler.fit_transform(ecg2[0 * fs * sleep_epoch_len:(0+i) * fs * sleep_epoch_len].reshape(1,-1)).T.squeeze(1), resamp_srate*sleep_epoch_len)) 
#             #     except:
#             #         import pdb;pdb.set_trace()
#             if index in train_ids:

#                 self.ecg_train2 = torch.cat([self.ecg_train2,torch.tensor(windowed_ecg2)], dim=0)
#                 self.eeg_trainc3a2 = torch.cat([self.eeg_trainc3a2,torch.tensor(windowed_eegc3a2)], dim=0)
#                 self.slp_stg_train = torch.cat([self.slp_stg_train,torch.tensor(slp_stg_coarse[0:num_windows])])
                
#             elif index in test_ids:

#                 self.ecg_test2 = torch.cat([self.ecg_test2,torch.tensor(windowed_ecg2)], dim=0)
#                 self.eeg_testc3a2 = torch.cat([self.eeg_testc3a2,torch.tensor(windowed_eegc3a2)], dim=0)
#                 self.slp_stg_test = torch.cat([self.slp_stg_test,torch.tensor(slp_stg_coarse[0:num_windows])])
                

#             self.n_slpstg += num_windows
        
#         ### Removing -2 labels from train data
#         self.ecg_train2 = self.ecg_train2[self.slp_stg_train != -2]
#         self.eeg_trainc3a2 = self.eeg_trainc3a2[self.slp_stg_train != -2]
#         self.slp_stg_train = self.slp_stg_train[self.slp_stg_train != -2]

#         #### SAVING TRAINING-EVAL DATA #####
#         split_indices = np.linspace(0, len(self.ecg_train2) - 1 , len(self.ecg_train2)) 
#         random.shuffle(split_indices)

#         self.ecg_training2 = self.ecg_train2[split_indices[: int(round(0.8 * len(self.ecg_train2)))].astype(int)]
#         self.eeg_trainingc3a2 = self.eeg_trainc3a2[split_indices[: int(round(0.8 * len(self.eeg_trainc3a2)))].astype(int)]
#         self.slp_stg_training = self.slp_stg_train[split_indices[: int(round(0.8 * len(self.ecg_train2)))].astype(int)]

#         import pdb;pdb.set_trace()

#         save_path = '/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/eeg_ecg_1ch/'
#         # WHEN USING BOTH EEG AND ECG CHANNELS USE STACK AND THEN SAVE
#         self.train_data = torch.stack([self.eeg_trainingc3a2,self.ecg_training2],dim=0)
#         torch.save(self.train_data, save_path + 'eeg_ecg_1ch_train.pt')
#         # torch.save(self.eeg_trainingc3a2, save_path + 'eeg1ch_train.pt')
#         torch.save(self.slp_stg_training, save_path + 'slp_stg_train_lbl.pt')


#         self.ecg_validation2 = self.ecg_train2[split_indices[int(round(0.8 * len(self.ecg_train2))):].astype(int)]
#         self.eeg_validationc3a2 = self.eeg_trainc3a2[split_indices[int(round(0.8 * len(self.eeg_trainc3a2))):].astype(int)]
#         self.slp_stg_validation = self.slp_stg_train[split_indices[int(round(0.8 * len(self.ecg_train2))):].astype(int)]

#         # WHEN USING BOTH EEG AND ECG CHANNELS
#         self.val_data = torch.stack([self.eeg_validationc3a2,self.ecg_validation2],dim=0)
#         torch.save(self.val_data, save_path + 'eeg_ecg_1ch_eval.pt')
#         # torch.save(self.eeg_validationc3a2, save_path + 'eeg1ch_eval.pt')
#         torch.save(self.slp_stg_validation, save_path + 'slp_stg_eval_lbl.pt')
        

#         ### Removing -2 labels from test data
#         self.ecg_test2 = self.ecg_test2[self.slp_stg_test != -2]
#         self.eeg_testc3a2 = self.eeg_testc3a2[self.slp_stg_test != -2]
#         self.slp_stg_test = self.slp_stg_test[self.slp_stg_test != -2]

#         #### SAVING  TEST DATA
#         # WHEN USING BOTH EEG AND ECG CHANNELS, STACK AND SAVE
#         self.test_data = torch.stack([self.eeg_testc3a2,self.ecg_test2],dim=0)
        
#         torch.save(self.test_data, save_path + 'eeg_ecg_1ch_test.pt')
#         # torch.save(self.eeg_test2, save_path + 'eeg1ch_test.pt')
#         torch.save(self.slp_stg_test, save_path + 'slp_stg_test_lbl.pt')
        
        
#     def __getitem__(self,index):
# #         # dataset indexing
#         return self.train_data[index],self.slp_stg_train[index],self.test_data[index],self.slp_stg_test[index]
#         # self.ecg_train[index],
#         # self.eeg_train[index],self.slp_stg_train[index],self.ecg_test[index],self.eeg_test[index],self.slp_stg_test[index]
        
        
#     def __len__(self):
#         return self.n_slpstg


class MassDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        cv=None,
        cv_idx=None,
        data_dir=None,
        eval_ratio=0.2,
        n_workers=12,
        n_jobs=-1,
        n_records=None,
        scaling="robust",
        adjustment=None,
        **kwargs,
    ):
        
        super().__init__()
        self.batch_size = batch_size
        self.n_workers= n_workers
        self.eval_ratio = eval_ratio
        self.data_dir = data_dir
        
#         MassDataset(self.data_dir,self.eval_ratio)
        
    def setup(self, stage = None):
        if stage == "fit":
            filepath = '/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/eeg_ecg_1ch/'
            self.train_data = torch.load(filepath + 'eeg_ecg_1ch_train.pt')
            self.train_slp_stg = torch.load(filepath + 'slp_stg_train_lbl.pt')
            # import pdb; pdb.set_trace()
            self.training_data = TensorDataset(
                # self.train_data,
                self.train_data[0],
                self.train_data[1],
                # self.train_data[2],
                self.train_slp_stg
                )
            self.val_data = torch.load(filepath + 'eeg_ecg_1ch_eval.pt')
            self.val_slp_stg = torch.load(filepath + 'slp_stg_eval_lbl.pt')
            # import pdb; pdb.set_trace()
            self.validation_data = TensorDataset(
                # self.val_data,
                self.val_data[0],
                self.val_data[1],
                # self.val_data[2],
                self.val_slp_stg
                )
        elif stage == "test":
            filepath = '/media/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/eeg_ecg_1ch/'
            self.test_data = torch.load(filepath + 'eeg_ecg_1ch_test.pt')
            self.test_slp_stg = torch.load(filepath + 'slp_stg_test_lbl.pt')
            # import pdb; pdb.set_trace()
            self.testing_data = TensorDataset(
                # self.test_data,
                self.test_data[0],
                self.test_data[1],
                # self.test_data[2],
                self.test_slp_stg
                )
    def train_dataloader(self):
        """Return training dataloader."""
        # import pdb;pdb.set_trace()
        self.setup("fit")
        return DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        self.setup("fit")
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        self.setup("test")
        return DataLoader(
            self.testing_data, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.n_workers
        )

data_path = "/media/acrophase/Sentinel_2/Dataset/Vaibhav/MASS/PT_FILES/eeg_ecg_1ch"
batch_size = 512

np.random.seed(42)
random.seed(42)

dm_params = dict(
    batch_size=batch_size,
    n_workers=12,
    data_dir=data_path,
    eval_ratio=0.2,
)
dm = MassDataModule(**dm_params)



import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.distributed as dist
from pytorch_lightning import LightningModule
import torchmetrics
from torchmetrics.functional import f1
import utils

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels=5, out_channels=5, kernel_size=3, dilation=1, activation="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation
        self.padding = (self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1) - 1) // 2

        self.layers = nn.Sequential(
            nn.ConstantPad1d(padding=(self.padding, self.padding), value=0),
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                bias=True,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.out_channels),
        )
        nn.init.xavier_uniform_(self.layers[1].weight)
        nn.init.zeros_(self.layers[1].bias)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, filters=[16, 32, 64, 128], in_channels=5, maxpool_kernels=[10, 8, 6, 4], kernel_size=5, dilation=2):
        super().__init__()
        self.filters = filters
        self.in_channels = in_channels
        self.maxpool_kernels = maxpool_kernels
        self.kernel_size = kernel_size
        self.dilation = dilation
        assert len(self.filters) == len(
            self.maxpool_kernels
        ), f"Number of filters ({len(self.filters)}) does not equal number of supplied maxpool kernels ({len(self.maxpool_kernels)})!"

        self.depth = len(self.filters)

        # fmt: off
        self.blocks = nn.ModuleList([nn.Sequential(
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                activation="relu",
            ),
            ConvBNReLU(
                in_channels=self.filters[k],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                activation="relu",
            ),
        ) for k in range(self.depth)])
        # fmt: on

        self.maxpools = nn.ModuleList([nn.MaxPool1d(self.maxpool_kernels[k]) for k in range(self.depth)])

        self.bottom = nn.Sequential(
            ConvBNReLU(
                in_channels=self.filters[-1],
                out_channels=self.filters[-1] * 2,
                kernel_size=self.kernel_size,
            ),
            ConvBNReLU(
                in_channels=self.filters[-1] * 2,
                out_channels=self.filters[-1] * 2,
                kernel_size=self.kernel_size
            ),
        )

    def forward(self, x):
        shortcuts = []
        for layer, maxpool in zip(self.blocks, self.maxpools):
            z = layer(x)
            shortcuts.append(z)
            x = maxpool(z)

        # Bottom part
        encoded = self.bottom(x)

        return encoded, shortcuts


class Decoder(nn.Module):
    def __init__(self, filters=[128, 64, 32, 16], upsample_kernels=[4, 6, 8, 10], in_channels=256, out_channels=5, kernel_size=5):
        super().__init__()
        self.filters = filters
        self.upsample_kernels = upsample_kernels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        assert len(self.filters) == len(
            self.upsample_kernels
        ), f"Number of filters ({len(self.filters)}) does not equal number of supplied upsample kernels ({len(self.upsample_kernels)})!"
        self.depth = len(self.filters)

        # fmt: off
        self.upsamples = nn.ModuleList([nn.Sequential(
            nn.Upsample(scale_factor=self.upsample_kernels[k]),
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
                activation='relu',
            )
        ) for k in range(self.depth)])

        self.blocks = nn.ModuleList([nn.Sequential(
            ConvBNReLU(
                in_channels=self.in_channels if k == 0 else self.filters[k - 1],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
            ),
            ConvBNReLU(
                in_channels=self.filters[k],
                out_channels=self.filters[k],
                kernel_size=self.kernel_size,
            ),
        ) for k in range(self.depth)])
        # fmt: off

    def forward(self, z, shortcuts):
        for upsample, block, shortcut in zip(self.upsamples, self.blocks, shortcuts[::-1]):
            z = upsample(z)
            z = torch.cat([shortcut, z], dim=1)
            z = block(z)

        return z

class SegmentClassifier(nn.Module):
    def __init__(self, sampling_frequency=200, num_classes=4, epoch_length=20):
        super().__init__()
        self.sampling_frequency = sampling_frequency
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.layers = nn.Sequential(
#             nn.AvgPool1d(kernel_size=(self.epoch_length * self.sampling_frequency)),
            # nn.Flatten(start_dim=2),
#             nn.ConstantPad1d(padding=(self.padding, self.padding), value=0),
            nn.Conv1d(in_channels=self.num_classes, out_channels=self.num_classes, kernel_size=1),
            nn.Softmax(dim=1),
        )
        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.zeros_(self.layers[0].bias)

    def forward(self, x):
        return self.layers(x)


class UTimeModel(LightningModule):
    # def __init__(
    #     self, filters=[16, 32, 64, 128], in_channels=5, maxpool_kernels=[10, 8, 6, 4], kernel_size=5,
    #     dilation=2, sampling_frequency=128, num_classes=5, epoch_length=30, lr=1e-4, batch_size=12,
    #     n_workers=0, eval_ratio=0.1, data_dir=None, n_jobs=-1, n_records=-1, scaling=None, **kwargs
    # ):
    def __init__(
        self,
        filters=None,
        in_channels=None,
        maxpool_kernels=None,
        kernel_size=None,
        dilation=None,
        num_classes=None,
        sampling_frequency=None,
        epoch_length=None,
        data_dir=None,
        n_jobs=None,
        lr=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(
            filters=self.hparams.filters,
            in_channels=self.hparams.in_channels,
            maxpool_kernels=self.hparams.maxpool_kernels,
            kernel_size=self.hparams.kernel_size,
            dilation=self.hparams.dilation,
        )
        self.decoder = Decoder(
            filters=self.hparams.filters[::-1],
            upsample_kernels=self.hparams.maxpool_kernels[::-1],
            in_channels=self.hparams.filters[-1] * 2,
            kernel_size=self.hparams.kernel_size,
        )
        self.dense = nn.Sequential(
            nn.Conv1d(in_channels=self.hparams.filters[0], out_channels=self.hparams.num_classes, kernel_size=1, bias=True),
            nn.Tanh()
        )
        nn.init.xavier_uniform_(self.dense[0].weight)
        nn.init.zeros_(self.dense[0].bias)
        self.segment_classifier = SegmentClassifier(
            sampling_frequency=self.hparams.sampling_frequency,
            num_classes=self.hparams.num_classes,
            epoch_length=self.hparams.epoch_length
        )
        self.loss = utils.DiceLoss(self.hparams.num_classes)

        # Create Dataset params
        self.dataset_params = dict(
            data_dir=self.hparams.data_dir,
            n_jobs=self.hparams.n_jobs,
        )

        # Create Optimizer params
        self.optimizer_params = dict(lr=self.hparams.lr)
        self.train_acc = torchmetrics.Accuracy()
        self.eval_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.train_cohenkappa = torchmetrics.CohenKappa(num_classes=self.hparams.num_classes)
        self.eval_cohenkappa = torchmetrics.CohenKappa(num_classes=self.hparams.num_classes)
        self.test_cohenkappa = torchmetrics.CohenKappa(num_classes=self.hparams.num_classes)


        self.train_conf_matrix = torchmetrics.ConfusionMatrix( num_classes=self.hparams.num_classes)
        self.eval_conf_matrix = torchmetrics.ConfusionMatrix( num_classes=self.hparams.num_classes)
        self.test_conf_matrix = torchmetrics.ConfusionMatrix( num_classes=self.hparams.num_classes)

    def forward(self, x):
        # Run through encoder
        z, shortcuts = self.encoder(x)

        # Run through decoder
        z = self.decoder(z, shortcuts)

        # Run dense modeling
        z = self.dense(z)

        return z

    def classify_segments(self, x, resolution=20):

        # Run through encoder + decoder
        z = self(x)
        # Classify decoded samples
        resolution_samples = self.hparams.sampling_frequency * resolution
        z = z.unfold(-1, resolution_samples, resolution_samples) \
             .mean(dim=-1)
        y = self.segment_classifier(z)

        return y
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), **self.optimizer_params
        )
    
    def training_step(self, batch_train, batch_idx):
        
        [eeg_train,ecg_train,y_train] = batch_train
        
        ## Choose modalities to train
        x_train = ecg_train.unsqueeze(1)
        # x_train = eeg_train.unsqueeze(1)
        # x_train = torch.cat((eeg_train.unsqueeze(1), ecg_train.unsqueeze(1)) ,dim = 1) 

        y_train = torch.nn.functional.one_hot(y_train.type(torch.int64), num_classes=self.hparams.num_classes).unsqueeze(1)
        pred_train = self.classify_segments(x_train.float(), resolution=self.hparams.epoch_length)
        loss_train,pred_train,y_train = self.compute_loss(pred_train, y_train)
        
        ## Metric ##
        self.train_acc(pred_train, y_train)
        self.train_cohenkappa(pred_train, y_train)
        self.train_conf_matrix(pred_train,y_train)
        self.train_f1 = f1(pred_train,y_train, num_classes=self.hparams.num_classes, average = 'weighted')
        ## Logging metrics
        self.log('train_loss', loss_train, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_CK', self.train_cohenkappa, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        # self.log('train_CFMat',self.train_conf_matrix,on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_F1_score',self.train_f1,on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return {
            'loss': loss_train,
            'predicted': pred_train,
        }
    
    def training_epoch_end(self, training_step_outputs):
        self.log('avg_train_loss', training_step_outputs[0]['loss'].mean())
    
    def validation_step(self, batch_eval, batch_idx):
        
        [eeg_val,ecg_val,y_val]= batch_eval
        
        ## Choose modalities to eval
        x_val = ecg_val.unsqueeze(1)
        # x_val = eeg_val.unsqueeze(1)
        # x_val = torch.cat((eeg_val.unsqueeze(1), ecg_val.unsqueeze(1)), dim = 1)

        y_val = torch.nn.functional.one_hot(y_val.type(torch.int64), num_classes=self.hparams.num_classes).unsqueeze(1)
        pred_val = self.classify_segments(x_val.float(), resolution=self.hparams.epoch_length)
        loss_val,pred_val,y_val = self.compute_loss(pred_val, y_val)

        ## Accuracy metric ##
        self.eval_acc(pred_val, y_val)
        self.eval_cohenkappa(pred_val, y_val)
        self.eval_conf_matrix(pred_val,y_val)
        self.eval_f1 = f1(pred_val,y_val, num_classes=self.hparams.num_classes, average = 'weighted')
        ## Logging metrics
        self.log('eval_loss', loss_val, on_epoch=True, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('eval_acc', self.eval_acc, on_epoch=True, on_step=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('eval_CK', self.eval_cohenkappa, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # self.log('eval_CFMat', self.eval_conf_matrix, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('eval_F1_score', self.eval_f1, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            
        return {
            'loss': loss_val,
            'predicted': pred_val,
        }
    
    def validation_epoch_end(self, val_step_outputs):
        self.log('avg_val_loss', val_step_outputs[0]['loss'].mean())
    
    def test_step(self, batch_test, batch_idx):
        
        [eeg_test,ecg_test, y_test] = batch_test
        
        ## Choose modalities to eval
        x_test=ecg_test.unsqueeze(1)
        # x_test=eeg_test.unsqueeze(1)
        # x_test = torch.cat((eeg_test.unsqueeze(1),ecg_test.unsqueeze(1)), dim = 1) 
        
        y_test = torch.nn.functional.one_hot(y_test.type(torch.int64), num_classes=4).unsqueeze(1)
        pred_test = self.classify_segments(x_test.float(), resolution=20)
        loss_test,pred_test, y_test = self.compute_loss(pred_test, y_test)
        
        ## Accuracy metric ##
        self.test_acc(pred_test, y_test)
        self.test_cohenkappa(pred_test, y_test)
        self.test_conf_matrix(pred_test, y_test)
        self.test_f1 = f1(pred_test, y_test, num_classes=self.hparams.num_classes, average = 'weighted')
        ## Logging metrics
        self.log('test_loss', loss_test, on_epoch=True, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_acc', self.test_acc, on_epoch=True, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test_CK', self.test_cohenkappa, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # self.log('test_CFMat', self.test_conf_matrix, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('test_F1_score', self.test_f1, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        y_1s = self.classify_segments(x_test.float(), resolution=1)
                
        return {
            'loss': loss_test,
            "predicted": pred_test,
            "true": y_test,
            'logits': y_1s
        }
    
    def test_epoch_end(self, test_step_outputs):
        self.log('avg_test_loss', test_step_outputs[0]['loss'].mean())
        
    
    def compute_loss(self, y_pred, y_true):

        if y_pred.shape[-1] != self.hparams.num_classes:
            y_pred = y_pred.permute(dims=[0, 2, 1])
        if y_true.shape[-1] != self.hparams.num_classes:
            y_true = y_true.permute(dims=[0, 2, 1])
        # return self.loss(y_pred, y_true.argmax(dim=-1))

        # return
        return self.loss(y_pred, y_true),y_pred, y_true

seed_everything(42)
model = UTimeModel(
    filters=[16, 32, 64, 128, 256], 
    in_channels=1, 
    maxpool_kernels=[2, 2, 2, 2, 2], 
    kernel_size=5,
    dilation=2, 
    sampling_frequency=200, 
    num_classes=4, 
    epoch_length=20, 
    lr=1e-3, 
    batch_size=batch_size,
    n_workers=12,
                )

checkpoint_callback = ModelCheckpoint(monitor='eval_loss',
                                    #   filename='MASS-SS4-{eval_loss:.2f}',
                                    #   dirpath='my/path/',
                                      save_top_k=3,
                                    #   save_weights_only=False,
                                      mode='min',
                                     )
# Most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
trainer = Trainer(
    callbacks=[checkpoint_callback], 
                    min_epochs=1, 
                    max_epochs=600,
                    check_val_every_n_epoch=2,
                    gpus=1,
    #               profiler="simple",
    #               plugins='ddp_sharded',
    #               auto_scale_batch_size='binsearch',
                  precision = 16,
#                     distributed_backend="ddp",
    progress_bar_refresh_rate=1
                    )

# trainer = Trainer(fast_dev_run=True,gpus =1 )


train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
test_dataloader = dm.test_dataloader()

trainer.fit(model, train_loader, val_loader)
trainer.test(model,test_dataloader)