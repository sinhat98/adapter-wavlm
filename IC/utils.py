import torch
import torch.nn as nn
import torchaudio
import numpy as np
import pandas as pd
import wandb
import random
from tqdm.notebook import tqdm

from transformers import (
    WavLMConfig, 
    Wav2Vec2FeatureExtractor,
)

SAMPLE_RATE = 16000
EXAMPLE_WAV_MAX_SEC = 10

class FluentCommandsDataset(object):
    def __init__(self, csv_file='train.csv'):
        self.df = pd.read_csv(csv_file)
        config=WavLMConfig.from_pretrained("superb/wav2vec2-large-superb-ic")
        self.id2label = config.id2label
        self.label2id = config.label2id
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-large-superb-ic")
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wav_path = self.df.loc[idx, 'wav']
        array, sr = torchaudio.load(wav_path)
        label = []

        for slot in ["action", "object", "location"]:
            value = self.df.loc[idx][slot]
            label.append(self.label2id[value])

        return {'file':wav_path, 'array':array, 'label':label}
    
class ICCollator(object):
    def __init__(self, extractor=None, lthresh=None):
        self.lthresh = lthresh
        self.extractor = extractor
        
    def __call__(self, batch):
        waveforms, targets = [], []
        for data in batch:
            if self.lthresh == None:
                waveforms += [data['array'].numpy().flatten()]
            else:
                waveforms += [data['array'].numpy().flatten()[:self.lthresh]]
            targets += [data['label']]
        targets = torch.LongTensor(targets)
        inputs = self.extractor(waveforms, sampling_rate=SAMPLE_RATE, padding=True, return_tensors='pt')

        sample = (inputs, targets)

        return sample
    
'''
    Example
    ----------------------------
    train_dataset = FluentCommandsDataset('/gs/hs1/tga-i/otake.s.ad/work/IC/fluent_speech_commands/train.csv')
    model_config = {
    'id2label': train_dataset.id2label,
    'label2id': train_dataset.label2id,
    'num_labels': len(train_dataset.id2label)
    }
    model = WavLMForSequenceClassification.from_pretrained('microsoft/wavlm-base-plus', **model_config)
'''

def train_model(model, dataloaders_dict, optimizer, scheduler, num_epochs, log_wandb=False):
    
    pbar_update = 1 / sum([len(v) for v in dataloaders_dict.values()])
    loss_fct = nn.CrossEntropyLoss()
    
    opt_flag = (type(optimizer) == list)
    sc_flag = (type(scheduler) == list)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    torch.backends.cudnn.benchmark = True
    
    act_idx=6; obj_idx=20; loc_idx=24
    
    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            for phase in ['train', 'val', 'test']:
                if phase == 'train':
                    model.train()
                elif phase == 'test':
                    if (epoch+1) % num_epochs:
                        for _ in range(len(dataloaders_dict[phase])):
                            pbar.update(pbar_update)
                        continue
                    model.eval()
                else:
                    model.eval()
                    
                epoch_loss = 0.0  
                epoch_corrects = 0.0 # for IC
                
                data_count = 0
                for inputs, labels in dataloaders_dict[phase]:
                    minibatch_size = inputs['input_values'].size(0)
                    data_count += minibatch_size
                    if opt_flag:
                        for opt in optimizer:
                            opt.zero_grad()
                    else:
                        optimizer.zero_grad()
                    
                    inputs = inputs.to(device)
                    logits = model(**inputs).logits.cpu()
                    
                    act_ids = labels[:,0]; logits_act = logits[:, :act_idx]
                    obj_ids = labels[:,1]; logits_obj = logits[:, act_idx:obj_idx]
                    loc_ids = labels[:,2]; logits_loc = logits[:, obj_idx:loc_idx]
                    
                
                    act_pred_ids = logits_act.argmax(dim=-1)
                    obj_pred_ids = logits_obj.argmax(dim=-1)
                    loc_pred_ids = logits_loc.argmax(dim=-1)
                    
                    preds = torch.stack([act_pred_ids, obj_pred_ids, loc_pred_ids], dim=1)
                    
                    act_corrects = act_pred_ids.squeeze().eq(act_ids)
                    obj_corrects = obj_pred_ids.squeeze().eq(obj_ids-act_idx)
                    loc_corrects = loc_pred_ids.squeeze().eq(loc_ids-obj_idx)
                    
                    corrects = torch.stack([act_corrects, obj_corrects, loc_corrects], dim=1)
                    epoch_corrects += corrects.prod(1).float().sum().item()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        if phase == 'train':
                            loss_act = loss_fct(logits_act, act_ids)
                            loss_obj = loss_fct(logits_obj, obj_ids-act_idx)
                            loss_loc = loss_fct(logits_loc, loc_ids-obj_idx)
                            loss = loss_act + loss_obj + loss_loc
                            loss.backward()
                            
                            if opt_flag:
                                for opt in optimizer:
                                    opt.step()
                            else:
                                optimizer.step()
                            loss_log = loss.item()
                            del loss
                            
                            epoch_loss += loss_log * minibatch_size
                            
                            if log_wandb:
                                wandb.log({'train/loss': loss_log})
                        
                        pbar.update(pbar_update)                

                epoch_acc = epoch_corrects / len(dataloaders_dict[phase].dataset)
                
                if phase=='train':
                    epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                    
                    if scheduler:
                        if sc_flag:
                            for sc in scheduler:
                                sc.step()
                        else:        
                            scheduler.step()
        
                    print('Epoch {}/{} | {:^5} |  Loss: {:.4f} ACC: {:.4f}'.format(
                        epoch+1,
                        num_epochs, 
                        phase, 
                        epoch_loss,
                        epoch_acc))
                    
                    if log_wandb:
                        wandb.log({'train/epoch':epoch+1,
                                   'train/epoch_loss':epoch_loss,
                                   'train/epoch_acc':epoch_acc
                                  })
                    
                else:
                    
                    print('Epoch {}/{} | {:^5} |  ACC: {:.4f}'.format(epoch+1,\
                                                                 num_epochs, phase, epoch_acc))
                    if log_wandb:
                        wandb.log({f'{phase}/epoch':epoch+1, 
                                   f'{phase}/epoch_acc':epoch_acc
                                  })
                    
    return model

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)