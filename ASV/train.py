import numpy as np
import pandas as pd
import torch, torchaudio
import torch.nn as nn

from transformers import Wav2Vec2FeatureExtractor

from utils import (
    TestDataset,
    TrainCollator,
    TestCollator,
    train_model,
    fix_seed,
)

from ..modeling import (
    AdaWavLMForSpeakerVerification,
    WavLMForSpeakerVerification
)



import numpy as np
import wandb
from sklearn.metrics import f1_score

from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.sampler import DynamicBatchSampler

pretrained_model= 'microsoft/wavlm-base-plus'

import argparse
from distutils.util import strtobool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='sample_run') 
    parser.add_argument('--use_skip', type=strtobool, default=False)
    parser.add_argument('--use_adapter_fc', type=strtobool, default=True)
    parser.add_argument('--use_adapter_norm', type=strtobool, default=True)
    parser.add_argument('--eadapter_act', type=str, default='relu')
    parser.add_argument('--ladapter_act', type=str, default='relu')
    parser.add_argument('--eada_emb_size', type=int, default=256)
    parser.add_argument('--lada_emb_size', type=int, default=512)
    parser.add_argument('--proj_size', type=int, default=768)
    parser.add_argument('--train_encada', type=strtobool, default=False)
    parser.add_argument('--train_encoder', type=strtobool, default=False)
    parser.add_argument('--train_lawithea', type=strtobool, default=False)
    parser.add_argument('--weighted_sum', type=strtobool, default=False) 
    parser.add_argument('--use_adapter_attn', type=strtobool, default=True)
    parser.add_argument('--use_adapter_ff', type=strtobool, default=True)
    parser.add_argument('--adapter_init_std', type=float, default=1e-3)
    parser.add_argument('--ladapter_init_std', type=float, default=1e-3)
    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    parser.add_argument('--eadapter_lr', type=float, default=1e-5)
    parser.add_argument('--ladapter_lr', type=float, default=1e-5)
    parser.add_argument('--down_lr', type=float, default=5e-4)
    parser.add_argument('--wandb_log', type=strtobool, default=False)

    args = parser.parse_args()

    if args.train_encoder or args.weighted_sum:
        model_config = {'num_labels': 1211,
                        'classifier_proj_size': args.proj_size,
                        'use_weighted_layer_sum': args.weighted_sum,
                        }

        learning_rate = {'projector':args.down_lr,
                        'classifier':args.down_lr,
                        'encoder':args.encoder_lr,
                        'layer_weight':args.down_lr,
                        'layer_norm':args.donw_lr,
                        }
        scheduler = {'type':'LambdaLR', 
                'step_down': [0.3, 0.5, 0.7, 1.0, 0.6, 0.3, 0],
                'step_encoder':[0.0, 0.3, 0.7, 1.0, 0.6, 0.3, 0]
                }
 
    elif args.train_encada:
        model_config = {'num_labels':1211,
                        'classifier_proj_size':args.proj_size,
                        'adapter_embedding_size': {str(i): args.eada_emb_size for i in range(0, 12)},
                        'use_adapter_attn':args.use_adapter_attn,
                        'use_adapter_ff':args.use_adapter_ff,
                        'eadapter_act': None if args.eadapter_act=='None' else args.eadapter_act,
                        }
        learning_rate = {'classifier':args.down_lr, 
                         'adapter':args.eadapter_lr,
                         'layer_norm':args.eadapter_lr
                        }

    elif args.train_lawithea:
        model_config = {'num_labels':1211,
                        'classifier_proj_size':args.proj_size,
                        'use_adapter_to_output':True,
                        'use_adapter_to_output_weighted_sum':True,
                        'adapter_to_output_layer_size': {str(i):args.lada_emb_size for i in range(0,12)},
                        'use_adapter_fc':args.use_adapter_fc,
                        'use_upsampling':args.use_skip,
                        'use_residual':args.use_skip,
                        'use_adapter_norm':args.use_adapter_norm,
                        'adapter_embedding_size': {str(i):args.eada_emb_size for i in range(0,11)},
                        'eadapter_act': None if args.eadapter_act=='None' else args.eadapter_act,
                        'ladapter_act': None if args.ladapter_act=='None' else args.ladapter_act,
                        'use_adapter_ff': True,
                        'use_adapter_attn': False,
                        'adapter_init_std': args.adapter_init_std
                    }
        learning_rate = {
            'classifier':args.down_lr,
            'adapter_to_output':args.ladapter_lr,
            'adapter_layer_weights':args.ladapter_lr, 
            'adapter_ff':args.eadapter_lr,
            'layer_norm':args.eadapter_lr,
            }
    
    else:
        model_config = {'num_labels':1211,
                        'classifier_proj_size':args.proj_size,
                        'use_adapter_to_output':True,
                        'adapter_to_output_layer_size': {str(i):args.lada_emb_size for i in range(0,12)},
                        'use_adapter_to_output_weighted_sum':True,
                        'use_residual':args.use_skip,
                        'use_upsampling':args.use_skip,
                        'ladapter_act': None if args.ladapter_act=='None' else args.ladapter_act,
                        'use_adapter_fc': args.use_adapter_fc,
                        'use_adapter_norm': args.use_adapter_norm
                        }
        learning_rate = {'classifier':args.down_lr, 
                         'adapter_to_output':args.ladapter_lr,
                         'adapter_layer_weights':args.ladapter_lr,
                         'layer_norm':args.ladapter_lr
                        }
    sc_step = [0.3, 0.5, 0.7, 1.0, 0.2, 0.1, 0]


    df = pd.read_csv('train.csv')
    df = df.drop('id', axis=1)
    data = df.to_dict(orient='record')
    data_dict = {i:data[i] for i in range(len(data))}

    dynamic_items = [
        {"func": lambda l: torchaudio.load(l)[0].flatten(),
        "takes": ["file"],
        "provides": "array"},
        {"func": lambda file: file,
        "takes": ["file"],
        "provides": "file"},
        {"func": lambda spk_id:spk_id,
        "takes": ["label"],
        "provides": "label"},
        {'func': lambda duration:duration,
         'takes':['duration'],
         'provides':'duration'},
    ]
    train_dataset_db = DynamicItemDataset(data_dict, dynamic_items)
    train_dataset_db.set_output_keys(["file", 'array', 'label', 'duration'])
    test_dataset = TestDataset('veri_test_id.csv')
    
    config={
            "pretrained_model": pretrained_model,
            "model_config": model_config,
            "epochs": 6,
            "batch_size": {'train':12*32,
                           'test':1},
            "learning_rate": learning_rate,
            'optimizer': 'RAdam' if args.train_encoder else 'Adam',
            "scheduler": scheduler if args.train_encoder or args.weighted_sum else {
                'type':'LambdaLR', 
                'step': sc_step
                } ,
            'lthresh':200000
        }

    if args.wandb_log:
        wandb.init(
            project="ASV",
            config=config,
            tags=[args.tag],
            id=args.run_name
        )


    # setting
    seed = config['seed']
    fix_seed(seed)
    num_epochs = config['epochs']
    batch_size = config['batch_size']
    sc_setting = config['scheduler']
    pretrained_model = config['pretrained_model']
    lthresh = config['lthresh']
    max_batch_len = batch_size['train']

    dynamic_batcher = DynamicBatchSampler(train_dataset_db,
            max_batch_length=max_batch_len,
            num_buckets= 1024,
            length_func=lambda x: x['duration'],
            shuffle=True,
            )

    extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
    train_collator = TrainCollator(extractor=extractor, lthresh=lthresh)
    test_collator = TestCollator(extractor=extractor)
    train_dataloader = torch.utils.data.DataLoader(train_dataset_db, batch_sampler=dynamic_batcher, collate_fn=train_collator, num_workers=8, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size['test'], shuffle=False, collate_fn=test_collator, num_workers=8, pin_memory=True)
    dataloader_dict = {'train': train_dataloader, 'test': test_dataloader}

    if args.train_encoder or args.weighted_sum:
        model = WavLMForSpeakerVerification.from_pretrained(pretrained_model, **model_config)
    else:
        model = AdaWavLMForSpeakerVerification.from_pretrained(pretrained_model, **model_config)

    encoder_param = []
    layernorm_param = []
    layerweight_param = []
    adapter_param = []
    adapter_to_output_param = []
    adapter_to_output_layer_weights_param=[]
    class_param = []
    proj_param = []
    pcount = 0
    adapcount = 0

    flag = True
    if args.train_encoder:
        layer_names = [str(i) for i in range(0, 12)]
    
    elif args.weighted_sum:
        layer_names = [str(i) for i in range(12)]
    
    elif args.train_encada:
        layer_names = ['layers.'+k for k in model_config["adapter_embedding_size"].keys()]
 
    else:
        layer_names = ['layers.'+k for k in model_config["adapter_to_output_layer_size"].keys()]

    for name, param in model.named_parameters():
        for layer in layer_names:
            if layer in name:
                flag=True
                break
            else:
                flag=False

        if 'classifier' in name:
            print('class_param: ', name)
            class_param.append(param)
            pcount += param.numel()

        elif 'projector' in name:
            print('proj_param: ', name)
            proj_param.append(param)
            pcount += param.numel()
            
        elif 'adapter_to_output_layer_weights' in name:
            adapter_to_output_layer_weights_param.append(param)
            print('adapter_to_output_layer_weights: ', name)
            pcount += param.numel(); adapcount += param.numel()

        elif ('encoder.layers' in name and 'layer_norm' in name and flag and not args.train_encoder):
            layernorm_param.append(param)
            print('layer_norm: ', name);pcount += param.numel()

        elif 'adapter_to_output' in name:
            adapter_to_output_param.append(param)
            print('adapter_output: ', name)
            pcount += param.numel(); adapcount+=param.numel()

        elif 'adapter_layer' in name:
            adapter_param.append(param)
            pcount += param.numel();print('adapter_layer: ', name)

        elif 'encoder.layers' in name and flag and args.train_encoder:
            encoder_param.append(param)
            pcount += param.numel();print('encoder: ', name)

        elif 'layer_weights' in name and args.weighted_sum:
            layerweight_param.append(param)
            print('layer_weight: ', name);pcount+=param.numel()

        else:
            print('frozen: ', name)
            param.requires_grad = False

    print(f'num of tuned params: {pcount}\n')
    config.update({'num_params(1e7)': pcount/1e7})
    config.update({'num_adapter_params': adapcount})

    if args.train_encoder:
        if args.use_common_sc:
            optimizer = torch.optim.Adam([
                {'params': class_param, 'lr': learning_rate['classifier']},
                {'params': proj_param, 'lr': learning_rate['classifier']},
                {'params': encoder_param, 'lr': learning_rate['encoder']},
            ])
            def func(epoch):
                return sc_setting['step'][epoch]
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, func)
            
        else:
            optimizer1 = torch.optim.RAdam([
                {'params': class_param, 'lr': learning_rate['classifier']},
                {'params': proj_param, 'lr': learning_rate['classifier']},
            ])
            optimizer2 = torch.optim.RAdam([
                {'params': encoder_param, 'lr': learning_rate['encoder']},
            ])
            optimizer = [optimizer1, optimizer2]
            def func1(epoch):
                return sc_setting['step_down'][epoch]
            def func2(epoch):
                return sc_setting['step_encoder'][epoch]
            scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer1, func1)
            scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, func2)
            scheduler = [scheduler1, scheduler2]
    

    elif args.weighted_sum:
        optimizer = torch.optim.Adam([
            {'params': class_param, 'lr': learning_rate['classifier']},
            {'params': proj_param, 'lr': learning_rate['classifier']},
            {'params': layerweight_param, 'lr': learning_rate['layer_weight']},
            {'params': layernorm_param, 'lr': learning_rate['layer_norm']},
        ])
        def func(epoch):
            return sc_setting['step_down'][epoch]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, func)
    
    elif args.train_encada:
        optimizer = torch.optim.Adam([
            {'params': class_param, 'lr': learning_rate['classifier']},
            {'params': proj_param, 'lr': learning_rate['classifier']},
            {'params': layernorm_param, 'lr': learning_rate['layer_norm']},
            {'params': adapter_param, 'lr':learning_rate['adapter']},
        ])
        def func(epoch):
            return sc_setting['step'][epoch]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)

    elif args.train_lawithea:
        optimizer = torch.optim.Adam([
            {'params': class_param, 'lr': learning_rate['classifier']},
            {'params': proj_param, 'lr': learning_rate['classifier']},
            {'params': layernorm_param, 'lr': learning_rate['layer_norm']},
            {'params': adapter_param, 'lr':learning_rate['adapter_ff']},
            {'params':adapter_to_output_layer_weights_param, 'lr':learning_rate['adapter_layer_weights']},
            {'params': adapter_to_output_param, 'lr': learning_rate['adapter_to_output']},
        ])
        def func(epoch):
            return sc_setting['step'][epoch]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)

    else:
        optimizer = torch.optim.Adam([
            {'params': class_param, 'lr': learning_rate['classifier']},
            {'params': proj_param, 'lr': learning_rate['classifier']},
            {'params': layernorm_param, 'lr': learning_rate['layer_norm']},
            {'params':adapter_to_output_layer_weights_param, 'lr':learning_rate['adapter_layer_weights']},
            {'params': adapter_to_output_param, 'lr': learning_rate['adapter_to_output']},
        ])
        def func(epoch):
            return sc_setting['step'][epoch]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
    model = train_model(model, dataloader_dict, optimizer, scheduler, num_epochs, val_interval=6, log_wandb=args.wandb_log)

    if not args.train_encada and not args.train_encoder and not args.weighted_sum:
        weight = torch.nn.functional.softmax(model.module.wavlm.encoder.adapter_to_output_layer_weights.detach().cpu()).numpy()
        result = {k: weight[i] for i,k in enumerate(layer_names)}
        print(result)

if __name__ == "__main__":
    main()
