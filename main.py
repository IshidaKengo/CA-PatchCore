import argparse
import torch
import os
import pytorch_lightning as pl

from model import MODEL      
from seg_image import Segmentation
from evaluate import evaluation

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_name', default='cad-sd') 
    parser.add_argument('--dataset_path', default=r'./dataset/Co-occurrence Anomaly Detection Screw Dataset') 

    parser.add_argument('--output_path', default=r'./outputs') #Directory to save the model outputs
    parser.add_argument('--evaluation_path', default=r'./evaluation') #Directory to save evaluation results
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--coreset_sampling_rate', default=0.01) #Sampling rate of greedy coreset subsampling
    parser.add_argument('--num_cluster', type=int, default=5) #Number of clusters in unsupervised segmentaition
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    
    ###Preparation of save directory###
    save_dir = os.path.join(args.output_path,f'num_clusters={args.num_cluster}')
    os.makedirs(save_dir, exist_ok=True)
    
    ###Unsupervised Segmentation###
    print('During Segmentaiton...')
    seg = Segmentation(args)
    seg.run()
    
    ###Run###
    print('Running...')
    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=save_dir, max_epochs=1, gpus = (device=="cuda") )
    model = MODEL(args=args, save_dir=save_dir)
    if args.phase == 'train':
        trainer.fit(model)
        trainer.test(model)
    elif args.phase == 'test':
        trainer.test(model)

    ###Evaluation###
    print('During Evaluation...')
    evaluation(args)
    
