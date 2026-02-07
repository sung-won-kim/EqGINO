import os
import time
import torch
import wandb
import pickle
import random
import argparse
import warnings
import importlib
import numpy as np
import lightning as L
from random import shuffle
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import DataLoader as PyGDataLoader
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

def main(args):
    if args.log_name == None :
        args.log_name = f'{args.model}_{args.data_fname}'

    for s_id, seed in enumerate(range(args.seed, args.seed + args.num_seed)):

        torch.manual_seed(seed)  #Torch
        random.seed(seed)        #Python
        np.random.seed(seed)     #NumPy
        L.seed_everything(seed, workers=True)

        module = importlib.import_module(f"model.{args.model}")
        eqgino = getattr(module, "EQGINO")
        LargeDataset = getattr(module, "LargeDataset")

        # _________
        # Load Data
        train_basepath = f'./data/{args.data_fname}/train'
        train_data_list = os.listdir(train_basepath)

        # Set 10% of train data as valid data
        shuffle(train_data_list)
        valid_data_list = train_data_list[:int(0.1 * len(train_data_list))]
        train_data_list = train_data_list[int(0.1 * len(train_data_list)):]

        test_basepath = f'./data/{args.data_fname}/test'
        test_data_list = os.listdir(test_basepath)

        train_data_list.sort()
        valid_data_list.sort()
        test_data_list.sort()

        class DataModule(L.LightningDataModule):
            def train_dataloader(self):
                train_dataset = LargeDataset(train_data_list,basepath=train_basepath, args=args)
                return PyGDataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

            def val_dataloader(self):
                val_dataset = LargeDataset(valid_data_list,basepath=train_basepath, args=args)
                return PyGDataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
            
            def test_dataloader(self):
                test_dataset = LargeDataset(test_data_list,basepath=test_basepath, args=args)
                return PyGDataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
            
            def predict_dataloader(self):
                test_dataset = LargeDataset(test_data_list,basepath=test_basepath, args=args)
                return PyGDataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
        # sample_data 
        with open(f'{train_basepath}/{train_data_list[0]}', 'rb') as f:
            sample_data = pickle.load(f)

        current_time = time.strftime("%m%d-%H%M")
        if args.log_name != None :
            logger = WandbLogger(project=f'{args.summary}_{args.data_fname}_{args.tgt_y}', name=f'{args.log_name}', config=args)  # W&B logger
        else :
            logger = WandbLogger(project=f'{args.summary}_{args.data_fname}_{args.tgt_y}', name=f'{args.model}', config=args)

        dirpath = f'./best_models/{args.summary}_{args.tgt_y}/{args.log_name}/s{seed}_{current_time}/'

        print(f'    # ========================================================================================== #')
        print(f'    # Model: {args.model}')
        print(f'    # Summary: {args.log_name}')
        print(f'    # Current Time: {current_time}')
        print(f'    # Dataset: {args.data_fname}')
        print(f'    # Seed: {seed}, Total: {s_id+1}/{args.num_seed}')
        print(f'    # ========================================================================================== #')


        checkpoint_callback = ModelCheckpoint(
            monitor='Valid RMSE',
            mode="min",
            dirpath=dirpath,
            filename='best',
            save_top_k=1,
            save_last=True)

        if not os.path.exists(dirpath): 
            os.makedirs(dirpath) 

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=args.devices,
            max_epochs=args.epochs,
            callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=5), LearningRateMonitor(logging_interval='epoch')],
            log_every_n_steps=1, 
            logger=logger,
            check_val_every_n_epoch=args.val_interval,
        )

        datamodule = DataModule()
    
        model = eqgino(sample_data, args)

        trainer.fit(model=model, datamodule=datamodule)

        best_path = trainer.checkpoint_callback.best_model_path

        # 2. Manually load it with the fix
        best_model = eqgino.load_from_checkpoint(
            best_path, 
            raw_sample_data=sample_data, 
            args=args, 
            map_location='cpu',
            weights_only=False  
        )

        # 3. Pass the explicitly loaded model instead of ckpt_path='best'
        trainer.test(model=best_model, datamodule=datamodule)

        wandb.finish() 

if __name__ == '__main__':

    timestr = time.strftime("%m$d")

    def list_of_ints(arg):
        if arg == 'cpu':
            return arg
        else:
            return list(map(int, arg.split(',')))
        
    def parse_args():
        parser = argparse.ArgumentParser()
        timestr = time.strftime("%m%d")

        parser.add_argument("--model", type=str, default='eqgino')
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--devices", type=list_of_ints, default='0')
        parser.add_argument("--lr", type=float, default=0.0005)
        parser.add_argument("--val_interval", type=int, default=1)
        parser.add_argument("--summary", type=str, default=f'{timestr}')
        parser.add_argument("--data_fname", type=str, default='ahmedbody', choices=['ahmedbody', 'shapenetcar'])
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--num_seed", type=int, default=1)
        parser.add_argument("--tgt_y", type=str, default='3d_ab_wss', choices=['3d_ab_k', '3d_ab_omega', '3d_ab_nut', '3d_ab_wss', '3d_ab_p', '3d_snc_press'])
        parser.add_argument("--log_name", type=str, default=None) 
        parser.add_argument("--gno_radius", type=float, default=0.1)
        parser.add_argument("--fno_n_layers", type=int, default=4)
        parser.add_argument("--fno_n_mode", type=int, default=32)
        parser.add_argument("--mesh_subsample_rate", type=float, default=10) 
        parser.add_argument("--mesh_subsample_rate_valid", type=float, default=1) 
        parser.add_argument("--num_workers", type=int, default=5)
        parser.add_argument("--num_groups", type=int, default=4)
        parser.add_argument("--aug_type", type=str, default='canonical', choices=['canonical', 'discrete', 'arbitrary'])

        return parser.parse_known_args()

    args, unknown = parse_args()

    main(args)
