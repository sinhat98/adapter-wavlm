import torch, torchaudio
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import wandb
import random

from torch.utils.data import BatchSampler
from tqdm.notebook import tqdm
from collections import namedtuple
from typing import List, Tuple, Union


PATH_DATA = '../data/voxceleb'


class TestDataset(object):
    def __init__(self, metafile='veri_test_id.csv'):
        self.df = pd.read_csv(metafile)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_id = self.df.loc[idx, 'file']
        path = os.path.join(PATH_DATA+'/wav', file_id)
        array, _ = torchaudio.load(path)

        sample = {
            'array': array,
            'fileID': file_id
        }
        return sample

class TrainCollator(object):
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
        targets = torch.tensor(targets)

        if self.extractor is not None:
            inputs = self.extractor(waveforms, sampling_rate=16000, return_tensors='pt', padding=True)
            sample = (inputs, targets)
        else:
            sample = (waveforms, targets)
        
        return sample
    

class TestCollator(object):
    def __init__(self, extractor=None, lthresh=None):
        self.lthresh = lthresh
        self.extractor = extractor
    def __call__(self, batch):
        waveforms, fileID = [], []

        for data in batch:
            if self.lthresh == None:
                waveforms += [data['array'].numpy().flatten()]
            else:
                waveforms += [data['array'].numpy().flatten()[:self.lthresh]]
            fileID.append(data['fileID'])
        
        if self.extractor is not None:
            inputs = self.extractor(waveforms, sampling_rate=16000, return_tensors='pt', padding=True)
            sample = (inputs, fileID)
        else:
            sample = (waveforms, fileID)
        
        return sample
    
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, all_speech, n_classes, n_samples):
        self.labels = np.array(labels)
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = all_speech
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
    
class Collator(object):
    def __init__(self, lthresh=None):
        self.lthresh = lthresh
    def __call__(self, batch):
        waveforms, targets = [], []

        for data in batch:
            if self.lthresh == None:
                waveforms += [data['array'].numpy().flatten()]
            else:
                waveforms += [data['array'].numpy().flatten()[:self.lthresh]]
            targets += [data['label']]
        targets = torch.tensor(targets)

        return waveforms, targets


def compute_eer(scores: Union[np.ndarray, List[float]],
                labels: Union[np.ndarray, List[int]]) -> Tuple[
                    float, float, np.ndarray, np.ndarray]:
    """Compute equal error rate(EER) given matching scores and corresponding labels
    Parameters:
        scores(np.ndarray,list): the cosine similarity between two speaker embeddings.
        labels(np.ndarray,list): the labels of the speaker pairs, with value 1 indicates same speaker and 0 otherwise.
    Returns:
        eer(float):  the equal error rate.
        thresh_for_eer(float): the thresh value at which false acceptance rate equals to false rejection rate.
        fr_rate(np.ndarray): the false rejection rate as a function of increasing thresholds.
        fa_rate(np.ndarray): the false acceptance rate as a function of increasing thresholds.
     """
    if isinstance(labels, list):
        labels = np.array(labels)
    if isinstance(scores, list):
        scores = np.array(scores)
    label_set = list(np.unique(labels))
    assert len(
        label_set
    ) == 2, f'the input labels must contains both two labels, but recieved set(labels) = {label_set}'
    label_set.sort()
    assert label_set == [
        0, 1
    ], 'the input labels must contain 0 and 1 for distinct and identical id. '
    eps = 1e-8
    #assert np.min(scores) >= -1.0 - eps and np.max(
    #    scores
    #  ) < 1.0 + eps, 'the score must be in the range between -1.0 and 1.0'
    same_id_scores = scores[labels == 1]
    diff_id_scores = scores[labels == 0]
    thresh = np.linspace(np.min(diff_id_scores), np.max(same_id_scores), 1000)
    thresh = np.expand_dims(thresh, 1)
    fr_matrix = same_id_scores < thresh
    fa_matrix = diff_id_scores >= thresh
    fr_rate = np.mean(fr_matrix, 1)
    fa_rate = np.mean(fa_matrix, 1)

    thresh_idx = np.argmin(np.abs(fa_rate - fr_rate))
    result = namedtuple('speaker', ('eer', 'thresh', 'fa', 'fr'))
    result.eer = (fr_rate[thresh_idx] + fa_rate[thresh_idx]) / 2
    result.thresh = thresh[thresh_idx, 0]
    result.fr = fr_rate
    result.fa = fa_rate

    return result


def train_model(model, dataloaders_dict, optimizer, scheduler, num_epochs, val_interval=6, log_wandb=False):
    df_clean = pd.read_csv('veri_test_cleaned.csv')
    idxs = np.random.choice(len(dataloaders_dict['train'].dataset), 10000)

    pbar_update = 1 / sum([len(v) for v in dataloaders_dict.values()])
    cossim = nn.CosineSimilarity(dim=-1)
    
    opt_flag = (type(optimizer) == list)
    sc_flag = (type(scheduler) == list)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 高速化
    torch.backends.cudnn.benchmark = True
    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            train_embs = []
            eval_phase = (epoch+1) // val_interval
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()
                else:
                    if not eval_phase:
                        for _ in range(len(dataloaders_dict[phase])):
                            pbar.update(pbar_update)
                        continue
                    model.eval()
                epoch_loss = 0.0  
                epoch_corrects = 0 # for SID
                emb_dict = {}
                
                data_count = 0
                for inputs, target in dataloaders_dict[phase]:
                    
                    minibatch_size = inputs['input_values'].size(0)
                    data_count += minibatch_size

                    if opt_flag:
                        for opt in optimizer:
                            opt.zero_grad()
                    else:
                        optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        
                        if phase == 'train':
                            inputs['labels'] = target
                            inputs = inputs.to(device)
                            outputs = model(**inputs)
                            loss = outputs.loss.mean(dim=-1)
                            preds = outputs.logits.argmax(dim=-1).to('cpu')
                            if eval_phase:
                                embeddings = outputs.embeddings.cpu().detach().clone()
                                for i in range(embeddings.shape[0]):
                                    train_embs.append(embeddings[i])
                            loss.backward()
                            if opt_flag:
                                for opt in optimizer:
                                    opt.step()
                            else:
                                optimizer.step()
                            loss_log = loss.item()
                            del loss
                            epoch_corrects += preds.squeeze().eq(target).sum().item()
                            epoch_loss += loss_log * minibatch_size
                            
                            if log_wandb:
                                wandb.log({'train/loss': loss_log})
                        else:
                            inputs = inputs.to(device)
                            outputs = model(**inputs)
                            embeddings = outputs.embeddings.cpu().detach().clone()
                            for i in range(len(target)):
                                emb_dict[target[i]] = embeddings[i]
                        
                        pbar.update(pbar_update)                
                
                if phase=='train':
                    epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                    epoch_acc = epoch_corrects / len(dataloaders_dict[phase].dataset)
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
                    train_cohort = torch.stack(train_embs)
                    sims_snorm_clean = []
                    sims_clean = []; inds_clean = []
                    
                    # s-norm
                    for i in range(len(df_clean)):
                        entry = df_clean.loc[i]
                        test = emb_dict[entry['file1']]
                        enrol = emb_dict[entry['file2']]
                        score = cossim(enrol, test).item()
                        inds_clean.append(entry['label'])
                        sims_clean.append(score)

                        enrol_rep = enrol.repeat(train_cohort[idxs].shape[0], 1)
                        score_e_c = cossim(enrol_rep, train_cohort[idxs])
                        mean_e_c = torch.mean(score_e_c, dim=0).item()
                        std_e_c = torch.std(score_e_c, dim=0).item()
                        
                        test_rep = test.repeat(train_cohort[idxs].shape[0], 1)
                        score_t_c = cossim(test_rep, train_cohort[idxs])
                        mean_t_c = torch.mean(score_t_c, dim=0).item()
                        std_t_c = torch.std(score_t_c, dim=0).item()
                        
                        score_e = (score - mean_e_c) / std_e_c
                        score_t = (score - mean_t_c) / std_t_c
                        score = 0.5 * (score_e + score_t)
                        sims_snorm_clean.append(score)
                        

                    epoch_eer_clean = compute_eer(sims_clean, inds_clean).eer
                    eer_clean = compute_eer(sims_snorm_clean, inds_clean).eer
                    print(f'EER (score-snorm) clean: {eer_clean:.4f}')

                    if log_wandb:
                        wandb.log({'eer':epoch_eer_clean,
                                   'eer_snorm': eer_clean})

    
    return model


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

