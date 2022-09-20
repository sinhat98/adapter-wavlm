import torch
import wandb
import argparse
import sys, os
sys.path.append(os.pardir)
from distutils.util import strtobool

from transformers import WavLMForSequenceClassification # Original WavLMModel
from modeling import AdaWavLMForSequenceClassification
from utils import (
    FluentCommandsDataset,
    ICCollator,
    train_model,
    fix_seed
)

fix_seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='sample_run') 

    parser.add_argument('--use_skip', type=strtobool, default=False)
    parser.add_argument('--use_adapter_fc', type=strtobool, default=True)
    parser.add_argument('--use_adapter_norm', type=strtobool, default=True)
    parser.add_argument('--eadapter_act', type=str, default='gelu')
    parser.add_argument('--ladapter_act', type=str, default='gelu')
    parser.add_argument('--eada_emb_size', type=int, default=256)
    parser.add_argument('--lada_emb_size', type=int, default=512)
    parser.add_argument('--proj_size', type=int, default=256)
    parser.add_argument('--use_adapter_attn', type=strtobool, default=True)
    parser.add_argument('--use_adapter_ff', type=strtobool, default=True)


    parser.add_argument('--train_encada', type=strtobool, default=False)
    parser.add_argument('--train_encoder', type=strtobool, default=False)
    parser.add_argument('--train_lawithea', type=strtobool, default=False)
    parser.add_argument('--weighted_sum', type=strtobool, default=False) 

    parser.add_argument('--adapter_init_std', type=float, default=1e-3)
    parser.add_argument('--ladapter_init_std', type=float, default=1e-3)
    parser.add_argument('--eadapter_lr', type=float, default=1e-5)
    parser.add_argument('--ladapter_lr', type=float, default=1e-5)

    parser.add_argument('--wandb_log', type=strtobool, default=False)

    args = parser.parse_args()

    train_dataset = FluentCommandsDataset('train.csv')
    val_dataset = FluentCommandsDataset('valid.csv')
    test_dataset = FluentCommandsDataset('test.csv')
    collator=ICCollator(train_dataset.extractor)

    if args.train_encoder or args.weighted_sum:
        model_config = {'id2label': train_dataset.id2label,
                        'label2id': train_dataset.label2id,
                        'num_labels': len(train_dataset.id2label),
                        'use_weighted_layer_sum': args.weighted_sum,
                        'classifier_proj_size':args.proj_size,
                    }

        learning_rate = {'down':5e-4,
                        'encoder':1e-4,
                        'layer_weight':args.ladapter_lr,
                        'layer_norm':args.ladapter_lr,
                        }

    elif args.train_encada:
        model_config = {'id2label': train_dataset.id2label,
                        'label2id': train_dataset.label2id,
                        'num_labels': len(train_dataset.id2label),
                        'classifier_proj_size':args.proj_size,
                        'adapter_embedding_size': {str(i): args.eada_emb_size for i in range(0, 12)},
                        'use_adapter_attn':args.use_adapter_attn,
                        'use_adapter_ff':args.use_adapter_ff,
                        'eadapter_act': None if args.eadapter_act=='None' else args.eadapter_act,
                        }
        learning_rate = {'down':5e-4, 
                         'adapter':args.eadapter_lr,
                         'layer_norm':args.eadapter_lr
                        }

    elif args.train_lawithea:
        model_config = {'id2label': train_dataset.id2label,
                        'label2id': train_dataset.label2id,
                        'num_labels': len(train_dataset.id2label),
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
                        'adapter_init_std': args.adapter_init_std,
                        'ladapter_init_std': args.ladapter_init_std,
                    }
        learning_rate = {
            'down':5e-4,
            'adapter_to_output':args.ladapter_lr,
            'adapter_layer_weights':args.ladapter_lr, 
            'adapter_ff':args.eadapter_lr,
            'layer_norm':args.eadapter_lr,
            }
    
    else:
        model_config = {'id2label': train_dataset.id2label,
                        'label2id': train_dataset.label2id,
                        'num_labels': len(train_dataset.id2label),
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
        learning_rate = {'down':5e-4, 
                         'adapter_to_output':args.ladapter_lr,
                         'adapter_layer_weights':args.ladapter_lr,
                         'layer_norm':args.ladapter_lr
                        }

    config={"pretrained_model": 'microsoft/wavlm-base-plus',
            "model_config": model_config,
            "epochs": 7,
            "batch_size": 16,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "scheduler": {'type':'LambdaLR', 'step': [0.1, 0.5, 0.7, 1.0, 0.5, 0.3, 0.1, 0]}
            }

    if args.wandb_log:
        wandb.init(
            project="IC",
            config=config,
            id=args.run_name
    )

    num_epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    sc_setting = config['scheduler']
    pretrained_model = config['pretrained_model']


    train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collator, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=collator, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=collator, batch_size=batch_size, shuffle=False, num_workers=8)
    dataloader_dict = {'train':train_loader, 'val':val_loader, 'test':test_loader}

    if args.train_encoder or args.weighted_sum:
        model = WavLMForSequenceClassification.from_pretrained(pretrained_model, **model_config)
    else:
        model = AdaWavLMForSequenceClassification.from_pretrained(pretrained_model, **model_config)

    down_param=[]
    encoder_param = []
    layernorm_param = []
    layerweight_param = []
    adapter_param = []
    adapter_to_output_param = []
    adapter_to_output_layer_weights_param=[]

    pcount = 0
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
        for idx in layer_names:
            if idx in name:
                flag = True; break
            else:
                flag = False

        if 'projector' in name or 'classifier' in name:
            print('down: ', name)
            pcount += param.numel()
            down_param.append(param)
        
        elif 'adapter_to_output_layer_weights' in name:
            adapter_to_output_layer_weights_param.append(param)
            print('adapter_to_output_layer_weights: ', name);pcount += param.numel()

        elif ('encoder.layers' in name and 'layer_norm' in name and flag and not args.train_encoder):
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
            
    print(f'num of tuned params: {pcount}\n')
    config.update({'num_params(1e7)':pcount/1e7})

    def func(epoch):
        return sc_setting['step'][epoch]

    if args.train_encoder:
        optimizer = torch.optim.Adam([
            {'params': down_param, 'lr': learning_rate['down']},
            {'params': encoder_param, 'lr': learning_rate['encoder']},
        ])            

    elif args.weighted_sum:
        optimizer = torch.optim.Adam([
            {'params': down_param, 'lr': learning_rate['down']},
            {'params': layerweight_param, 'lr': learning_rate['layer_weight']},
            {'params': layernorm_param, 'lr': learning_rate['layer_norm']},
        ])
    
    elif args.train_encada:
        optimizer = torch.optim.Adam([
            {'params': down_param, 'lr': learning_rate['down']},
            {'params': layernorm_param, 'lr': learning_rate['layer_norm']},
            {'params': adapter_param, 'lr':learning_rate['adapter']},
        ])

    elif args.train_lawithea:
        optimizer = torch.optim.Adam([
            {'params': down_param, 'lr': learning_rate['down']},
            {'params': layernorm_param, 'lr': learning_rate['layer_norm']},
            {'params': adapter_param, 'lr':learning_rate['adapter_ff']},
            {'params':adapter_to_output_layer_weights_param, 'lr':learning_rate['adapter_layer_weights']},
            {'params': adapter_to_output_param, 'lr': learning_rate['adapter_to_output']},
        ])

    else:
        optimizer = torch.optim.Adam([
            {'params': down_param, 'lr': learning_rate['down']},
            {'params': layernorm_param, 'lr': learning_rate['layer_norm']},
            {'params':adapter_to_output_layer_weights_param, 'lr':learning_rate['adapter_layer_weights']},
            {'params': adapter_to_output_param, 'lr': learning_rate['adapter_to_output']},
        ])
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, func)
    
    model = train_model(model,
                        dataloader_dict,
                        optimizer, 
                        scheduler, 
                        num_epochs, 
                        log_wandb=args.wandb_log
                    )

    if not args.train_encada and not args.train_encoder and not args.weighted_sum:
        weight = torch.nn.functional.softmax(model.module.wavlm.encoder.adapter_to_output_layer_weights.detach().cpu()).numpy()
        result = {k: weight[i] for i,k in enumerate(layer_names)}
        print(result)

if __name__ == '__main__':
    main()
