import os
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
import wandb
import random
from tqdm.notebook import tqdm


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


## Dataset
class IemocapDataset(object):
    """
        Create a Dataset for Iemocap. Each item is a tuple of the form:
        (waveform, sample_rate, emotion, activation, valence, dominance)
    """

    _ext_audio = '.wav'
    _emotions = { 'ang': 0, 'hap': 1, 'exc': 1, 'sad': 3, 'fru': 4, 'fea': 5, 'sur': 6, 'neu': 7, 'xxx': 8 }
    def __init__(self,
                 root='IEMOCAP_full_release',
                 emotions=['ang', 'hap', 'exc', 'sad', 'neu'],
                 sessions=[1, 2, 3, 4, 5],
                 script_impro=['script', 'impro'],
                 genders=['M', 'F'],
                 emapping=None):
        """
        Args:
            root (string): Directory containing the Session folders
        """
        self.root = root

        # Iterate through all 5 sessions
        data = []
        for i in range(1, 6):
            # Define path to evaluation files of this session
            path = os.path.join(root, 'Session' + str(i), 'dialog', 'EmoEvaluation')

            # Get list of evaluation files
            files = [file for file in os.listdir(path) if file.endswith('.txt')]

            # Iterate through evaluation files to get utterance-level data
            for file in files:
                # Open file
                f = open(os.path.join(path, file), 'r')

                # Get list of lines containing utterance-level data. Trim and split each line into individual string elements.
                data += [line.strip()
                             .replace('[', '')
                             .replace(']', '')
                             .replace(' - ', '\t')
                             .replace(', ', '\t')
                             .split('\t')
                         for line in f if line.startswith('[')]

        # Get session number, script/impro, speaker gender, utterance number
        data = [d + [d[2][4], d[2].split('_')[1], d[2][-4], d[2][-3:]] for d in data]

        # Create pandas dataframe
        self.df = pd.DataFrame(data, columns=['start', 'end', 'file', 'emotion', 'activation', 'valence', 'dominance', 'session', 'script_impro', 'gender', 'utterance'], dtype=np.float32)

        # Filter by emotions
        filtered_emotions = self.df['emotion'].isin(emotions)
        self.df = self.df[filtered_emotions]

        # Filter by sessions
        filtered_sessions = self.df['session'].isin(sessions)
        self.df = self.df[filtered_sessions]

        # Filter by script_impro
        filtered_script_impro = self.df['script_impro'].str.contains('|'.join(script_impro))
        self.df = self.df[filtered_script_impro]

        # Filter by gender
        filtered_genders = self.df['gender'].isin(genders)
        self.df = self.df[filtered_genders]

        # Reset indices
        self.df = self.df.reset_index()

        # Map emotion labels to numeric values
        if emapping is not None:
            self.df['emotion'] = self.df['emotion'].map(emapping).astype(np.float32)
        else:
            self.df['emotion'] = self.df['emotion'].map(self._emotions).astype(np.float32)

        # Map file to correct path w.r.t to root
        self.df['file'] = [os.path.join('Session' + file[4], 'sentences', 'wav', file[:-5], file + self._ext_audio) for file in self.df['file']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.root, self.df.loc[idx, 'file'])
        waveform, sample_rate = torchaudio.load(audio_name)
        emotion = self.df.loc[idx, 'emotion']
        activation = self.df.loc[idx, 'activation']
        valence = self.df.loc[idx, 'valence']
        dominance = self.df.loc[idx, 'dominance']

        sample = {
            'path': audio_name,
            'waveform': waveform,
            'sample_rate': sample_rate,
            'emotion': emotion,
            'activation': activation,
            'valence': valence,
            'dominance': dominance
        }

        return sample

# Collator
class Collator(object):
    def __init__(self, extractor=None, lthresh=None):
        self.lthresh = lthresh
        self.extractor = extractor
    def __call__(self, batch):
        waveforms, targets = [], []
        files = []
        for data in batch:
            if self.lthresh == None:
                waveforms += [data['waveform'].numpy().flatten()]
            else:
                waveforms += [data['waveform'].numpy().flatten()[:self.lthresh]]
            targets += [torch.tensor(int(data['emotion']))]
            files += [data['path']]
        targets = torch.stack(targets)
        sampling_rate = self.extractor.sampling_rate
        inputs = self.extractor(waveforms, sampling_rate=sampling_rate, padding=True, return_tensors='pt')
        
        sample = (inputs, targets)
        
        return sample


# for trainning model and enference
def train_model(model, extractor, dataloaders_dict, optimizer, scheduler, num_epochs, val_interval=1, wandb_log=True):
    
    num_labels = model.config.num_labels
    pbar_update = 1 / sum([len(v) for v in dataloaders_dict.values()])
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    torch.backends.cudnn.benchmark = True

    with tqdm(total=num_epochs) as pbar:
        
        val_wa = [] # weighted accuracy
        val_ua = [] # unweighted accuracy

        for epoch in range(num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  
                else:
                    if (epoch+1) % val_interval:
                        for _ in range(len(dataloaders_dict[phase])):
                            pbar.update(pbar_update)
                        continue
                    model.eval()   
                epoch_loss = 0.0  
                epoch_corrects = 0
                class_corrects = np.zeros(num_labels)
                target_counts = np.zeros(num_labels)

                for step, (inputs, target) in enumerate(dataloaders_dict[phase]):
                    
                    bs = target.shape[0]

                    inputs = inputs.to(device)
                    target = target.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(**inputs, labels=target)
                        
                        logits = outputs.logits
                        loss = outputs.loss
                        
                        loss = loss.mean(dim=-1)
                        preds = torch.argmax(logits, dim=-1) 

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            loss_log = loss.item()
                            del loss
                            if wandb_log:
                                wandb.log({'train/loss':loss_log})
                        else:
                            for p, t in zip(preds, target):
                                target_counts[t] += 1
                                if p==t:
                                    class_corrects[t] += 1
                       
                        epoch_loss += loss_log * bs
                        epoch_corrects += preds.squeeze().eq(target).sum().item()
                        pbar.update(pbar_update)


                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                epoch_acc = epoch_corrects / len(dataloaders_dict[phase].dataset)
                
                if phase=='train':
                    if scheduler:
                        scheduler.step()
                    print('Epoch {}/{} | {:^5} |  Loss: {:.4f} UA: {:.4f}'.format(epoch+1,
                                                                                  num_epochs, 
                                                                                  phase, 
                                                                                  epoch_loss, 
                                                                                  epoch_acc))
                    if wandb_log:
                        wandb.log({'train/epoch':epoch+1,\
                                   'train/epoch_loss':epoch_loss, 
                                   'train/epoch_acc':epoch_acc}
                                )
                else:
                    val_ua.append(epoch_acc)
                    epoch_wacc = (class_corrects / target_counts).mean()
                    val_wa.append(epoch_wacc)
                    print('Epoch {}/{} | {:^5} |  Loss: {:.4f} UA: {:.4f} WA: {:.4f}'.format(epoch+1,
                                                                                             num_epochs, 
                                                                                             phase, 
                                                                                             epoch_loss, 
                                                                                             epoch_acc, 
                                                                                             epoch_wacc))
                    if wandb_log:
                        wandb.log({'val/epoch':epoch+1,
                                   'val/epoch_loss':epoch_loss,
                                   'val/epoch_UA':epoch_acc,
                                   'val/epoch_WA':epoch_wacc})
    
    print('max_valUA: ', max(val_ua))
    print('max_valWA: ', max(val_wa))    
    
    outputs = {'model':model, 'ua':val_ua[-1], 'wa':val_wa[-1]}
    
    return outputs