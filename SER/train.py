import torch
import numpy as np
import wandb
import argparse
from torch.utils.data import DataLoader

from utils import (
    IemocapDataset, 
    Collator, 
    train_model, 
    fix_seed
)

from transformers import (
    Wav2Vec2FeatureExtractor,
    WavLMForSequenceClassification # Original WavLMModel
)

from ..modeling import AdaWavLMForSequenceClassification # WavLM with Adapter

from sklearn.model_selection import train_test_split, KFold

from distutils.util import strtobool

extractor = Wav2Vec2FeatureExtractor.from_pretrained('superb/wav2vec2-base-superb-er') # for ER
train_collator = Collator(extractor, 200000) 
test_collator = Collator(extractor)

pretrained_model = 'wavlm-base-plus'
seed=42; fix_seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    parser.add_argument('--use_skip', type=strtobool, default=False)
    parser.add_argument('--use_adapter_fc', type=strtobool, default=True)
    parser.add_argument('--use_adapter_norm', type=strtobool, default=True)
    parser.add_argument('--eadapter_act', default='relu')
    parser.add_argument('--ladapter_act', default='relu')
    parser.add_argument('--lada_emb_size', type=int, default=512)
    parser.add_argument('--eada_emb_size', type=int, default=256)
    parser.add_argument('--proj_size', type=int, default=256)
    parser.add_argument('--train_encada', type=strtobool, default=False)
    parser.add_argument('--train_encoder', type=strtobool, default=False)
    parser.add_argument('--train_lawithea', type=strtobool, default=False)
    parser.add_argument('--weighted_sum', type=strtobool, default=False) 
    parser.add_argument('--use_adapter_attn', type=strtobool, default=True)
    parser.add_argument('--use_adapter_ff', type=strtobool, default=True)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--adapter_init_std', type=float, default=1e-3)
    parser.add_argument('--ladapter_init_std', type=float, default=1e-3)
    parser.add_argument('--downhead_lr', type=float, default=5e-4)
    parser.add_argument('--encoder_lr', type=float, default=5e-5)
    parser.add_argument('--ladapter_lr', type=float, default=1e-4)
    parser.add_argument('--eadapter_lr', type=float, default=1e-5)
    parser.add_argument('--wandb_log', type=strtobool, default=False)
    args = parser.parse_args()

    if args.train_encoder or args.weighted_sum:
        model_config = {'num_labels':4,
                        'classifier_proj_size':args.proj_size,
                        'use_weighted_layer_sum':args.weighted_sum
                        }
        learning_rate = {'classifier': args.downhead_lr,
                         'encoder': args.encoder_lr,
                         'layer_weight': args.ladapter_lr,
                         'layer_norm': args.ladapter_lr}

    elif args.train_encada:
        model_config = {'num_labels':4,
                        'classifier_proj_size':args.proj_size,
                        'adapter_embedding_size': {str(i): args.eada_emb_size for i in range(0, 12)}, # insert into all layers
                        'use_adapter_attn':args.use_adapter_attn,
                        'use_adapter_ff':args.use_adapter_ff,
                        'eadapter_act': args.eadapter_act
                        }
        learning_rate = {'classifier':args.downhead_lr, 
                         'adapter':args.eadapter_lr,
                         'layer_norm':args.eadapter_lr
                        }

    elif args.train_lawithea:
        model_config = {'num_labels':4,
                        'classifier_proj_size':args.proj_size,
                        'use_adapter_to_output':True,
                        'adapter_to_output_layer_size': {str(i):args.lada_emb_size for i in range(0, 12)} if args.new_exp\
                                                            else {str(i):args.lada_emb_size for i in range(0,12)},
                        'use_adapter_to_output_weighted_sum':True,
                        'adapter_embedding_size': {str(i):args.eada_emb_size for i in range(0,11)},
                        'use_residual':args.use_skip,
                        'use_upsampling':args.use_skip,
                        'adapter_init_std': args.adapter_init_std,
                        'eadapter_act': None if args.eadapter_act=='None' else args.eadapter_act,
                        'ladapter_act': None if args.ladapter_act=='None' else args.ladapter_act,
                        'adapter_init_std': args.adapter_init_std,
                        'ladapter_init_std': args.ladapter_init_std,
                        'use_adapter_fc': args.use_adapter_fc,
                        'use_adapter_norm': args.use_adapter_norm,
                        'use_adapter_attn': False,
                        'use_adapter_ff': True,
                        }
        learning_rate = {'classifier':args.downhead_lr,
                         'adapter_to_output':args.ladapter_lr,
                         'adapter_layer_weights':args.ladapter_lr,
                         'adapter': args.eadapter_lr,
                         'layer_norm':args.eadapter_lr
                        }

    else:
        model_config = {'num_labels':4,
                        'classifier_proj_size':args.proj_size,
                        'use_adapter_to_output':True,
                        'adapter_to_output_layer_size': {str(i):args.lada_emb_size for i in range(0,12)},
                        'use_adapter_to_output_weighted_sum':True,
                        'use_residual':args.use_skip,
                        'use_upsampling':args.use_skip,
                        'ladapter_init_std': args.ladapter_init_std,
                        'ladapter_act': None if args.ladapter_act=='None' else args.ladapter_act,
                        'use_adapter_fc': args.use_adapter_fc,
                        'use_adapter_norm': args.use_adapter_norm
                        }
        learning_rate = {'classifier':args.downhead_lr, 
                         'adapter_to_output':args.ladapter_lr,
                         'adapter_layer_weights':args.ladapter_lr,
                         'layer_norm':args.ladapter_lr
                        }

    config={
        "pretrained_model": pretrained_model,
        "model_config": model_config,
        "dataset": 'IEMOCAP_full',
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": learning_rate,
        "scheduler": {'type':'StepLR', 'step':10, 'gamma':0.1},
    }

    if args.wandb_log:
        wandb.init(
            project="SER",
            config=config,
            id=args.run_name,
            settings=wandb.Settings(start_method='fork')
        )
        config = wandb.config

    # 設定
    num_epochs = config['epochs']
    batch_size = config['batch_size']

    _emotions = {'ang': 0, 'hap': 1, 'exc': 1, 'sad': 2, 'neu':3}
    dataset = IemocapDataset(root='../data/IEMOCAP_full_release', script_impro=['script', 'impro'], emapping=_emotions)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    ua = []
    wa = []
    for k, (train_indices, val_indices) in enumerate(kf.split(range(len(dataset)))):
        print(f'Split: {k}')

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collator, num_workers=12, pin_memory=True)
        test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_collator, num_workers=12, pin_memory=True)
        dataloaders_dict = {'train': train_loader, 'val': test_loader}

        if args.train_encoder or args.weighted_sum:
            model = WavLMForSequenceClassification.from_pretrained(pretrained_model, **model_config)
        else:
            model = AdaWavLMForSequenceClassification.from_pretrained(pretrained_model, **model_config)

        down_param = []
        layernorm_param = []
        layerweight_param = []
        adapter_to_output_layer_weights_param = []
        adapter_to_output_param = []
        adapter_param = []
        encoder_param = []
        pcount = 0
        if args.train_encoder:
            layer_names = [str(i) for i in range(0, 12)]
        elif args.weighted_sum:
            layer_names = [str(i) for i in range(12)]
        elif args.train_encada:
            layer_names = ['layers.'+k for k in model_config["adapter_embedding_size"].keys()]
        else:
            layer_names = ['layers.'+k for k in model_config["adapter_to_output_layer_size"].keys()]
        flag = True
        for name, param in model.named_parameters():
            for layer in layer_names:
                if layer in name:
                    flag=True
                    break
                else:
                    flag=False

            if 'projector' in name or 'classifier' in name:
                print('down_param: ', name)
                pcount += param.numel()
                down_param.append(param)

            elif 'adapter_to_output_layer_weights' in name:
                adapter_to_output_layer_weights_param.append(param)
                print('adapter_to_output_layer_weights: ', name);pcount += param.numel()

            elif 'encoder.layers' in name and 'layer_norm' in name and flag and not args.train_encoder:
                layernorm_param.append(param)
                print('layer_norm: ', name);pcount += param.numel()

            elif 'adapter_to_output' in name:
                adapter_to_output_param.append(param)
                pcount += param.numel();print('adapter_output: ', name)

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


        print('\ncount of parameters: ', pcount, '\n')
        config.update({'num_params (1e7)': pcount/1e7})
        
        if args.train_encoder:
            optimizer = torch.optim.Adam([
                    {'params': down_param, 'lr': learning_rate['classifier']},
                    {'params': encoder_param, 'lr':learning_rate['encoder']}
                ])

        elif args.weighted_sum:
            optimizer = torch.optim.Adam([
                {'params': down_param, 'lr': learning_rate['classifier']},
                {'params': layerweight_param, 'lr':learning_rate['layer_weight']},
                {'params': layernorm_param, 'lr':learning_rate['layer_norm']}
            ])                

        elif args.train_encada:
            optimizer = torch.optim.Adam([
                {'params': down_param, 'lr': learning_rate['classifier']},
                {'params': layernorm_param, 'lr': learning_rate['layer_norm']},
                {'params':adapter_param, 'lr':learning_rate['adapter']},
            ])

        elif args.train_lawithea:
            optimizer = torch.optim.Adam([
                {'params': down_param, 'lr': learning_rate['classifier']},
                {'params': layernorm_param, 'lr': learning_rate['layer_norm']},
                {'params':adapter_to_output_layer_weights_param, 'lr':learning_rate['adapter_layer_weights']},
                {'params': adapter_to_output_param, 'lr': learning_rate['adapter_to_output']},
                {'params':adapter_param, 'lr':learning_rate['adapter']},
            ])

        else:
            optimizer = torch.optim.Adam([
                {'params': down_param, 'lr': learning_rate['classifier']},
                {'params': layernorm_param, 'lr': learning_rate['layer_norm']},
                {'params':adapter_to_output_layer_weights_param, 'lr':learning_rate['adapter_layer_weights']},
                {'params': adapter_to_output_param, 'lr': learning_rate['adapter_to_output']},
                ])

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler']['step'], gamma=config['scheduler']['gamma'])

        outputs = train_model(model, extractor, dataloaders_dict, optimizer, scheduler, num_epochs, step_log=False, val_interval=args.val_interval)
        model = outputs['model']

        if not args.train_encada and not args.train_encoder and not args.weighted_sum:        
            weight = torch.nn.functional.softmax(model.module.wavlm.encoder.adapter_to_output_layer_weights.detach().cpu()).numpy()
            result = {k: weight[i] for i,k in enumerate(layer_names)}
            print(result, '\n')

        uacc = outputs['ua']; wacc = outputs['wa']
        print(f'ua: {uacc}, wa: {wacc}')
        ua.append(uacc)
        wa.append(wacc)
        torch.cuda.empty_cache()
        del model

    ua = np.array(ua)
    wa = np.array(wa)

    print(f'UA: {ua.mean():.4f} +/- {ua.std():.4f}')
    print(f'WA: {wa.mean():.4f} +/- {wa.std():.4f}')

    config.update({'result': f'WA: {wa.mean():.4f} +/- {wa.std():.4f}'})
    
if __name__=='__main__':
    main()
