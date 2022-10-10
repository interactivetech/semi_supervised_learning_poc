
import determined as det
from model import wrn_28_2
import torch
from tqdm import tqdm
import torch.nn.functional as F
from ema import EMA,EMADriver, set_ema_model
from train import trainer
from eval import predict, eval
from tqdm import tqdm
from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer
from semilearn.datasets.cv_datasets import get_cifar
import argparse
from semilearn.core.utils import get_dataset, get_data_loader, get_optimizer, get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from flexmatch import FlexMatch


def set_args(args_d):
    
    
    parser = argparse.ArgumentParser(description='Semi-Supervised Learning (USB semilearn package)')
    args = parser.parse_args("")
    # args
    for k in args_d:
            setattr(args, k, args_d[k])
    
    return args

def get_models(device):
    '''
    '''
    m = wrn_28_2(pretrained=False,pretrained_path=None ,num_classes=10)
    _=m.to(device)
    ema = wrn_28_2(pretrained=False,pretrained_path=None ,num_classes=10)
    ema = set_ema_model(ema, m)
    emaA = EMADriver(model=m,ema_model=ema,ema_m=0.999)
    # emaA = EMADriver(model=m,ema_model=ema,ema_m=0.999)
    # emaA.before_run()
    return m, ema, emaA
def trainer():
    '''
    '''
    return

def main(hparams):
    '''
    '''
    args_d = {'dataset': 'cifar10',
         'num_classes': 10,
         'train_sampler': 'RandomSampler',
         'num_workers': 8,
         'lb_imb_ratio': 1,
         'ulb_imb_ratio':1.0,
          'batch_size': 32,
         'ulb_num_labels': 150,
         'img_size': 32,
         'crop_ratio': 0.875,
         'num_labels': 30,
         'seed': 1,
         'epoch': 3,
         'num_train_iter':150,
         'net': 'wrn_28_8',
         'optim': 'SGD',
         'lr': 0.03,
         'momentum': 0.9,
         'weight_decay': 0.0005,
         'layer_decay': 0.75,
          'num_warmup_iter': 0,
         'algorithm': None,
         'data_dir': './data',
         'uratio': 3,
         'eval_batch_size': 64}
    args = set_args(args_d)
    
    dataset_dict = get_dataset(args, 
                           args.algorithm, 
                           args.dataset, 
                           args.num_labels, 
                           args.num_classes, 
                           data_dir=args.data_dir,
                          include_lb_to_ulb=False)
    train_lb_loader = get_data_loader(args, dataset_dict['train_lb'], args.batch_size)
    train_lb_loader = get_data_loader(args, dataset_dict['train_lb'], args.batch_size)
    train_ulb_loader = get_data_loader(args, dataset_dict['train_ulb'], int(args.batch_size * args.uratio))
    eval_loader = get_data_loader(args, dataset_dict['eval'], args.eval_batch_size)
    device = torch.device('cuda:0')

    m,ema,emaA = get_models(device)
    emaA.before_run()
    

    optimizer = get_optimizer(m, args.optim, args.lr, args.momentum, args.weight_decay, args.layer_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                args.num_train_iter,
                                                num_warmup_steps=args.num_warmup_iter)
    loss_ce = torch.nn.CrossEntropyLoss()
    
    f = FlexMatch(T=1.0, 
             p_cutoff=0.95, 
             ulb_dest_len=len(dataset_dict['train_ulb']),
             num_classes=10,
             model=m,
             ema_model=ema,
             loss_ce=loss_ce,
             scheduler=scheduler,
             optimizer=optimizer,
             device=device,
             train_lb_loader=train_lb_loader,
             train_ulb_loader=train_ulb_loader,
             ulb_loss_ratio=1.0,
             hard_label=True, 
             thresh_warmup=True)
    
    steps, sup_loss,unsup_loss,total_loss = f.fit()
    # trainer()

if __name__ == '__main__':
    # NEW - defining distributed option for det.core.init()
    DID_DET_SUCCEED=False
# try:
    info = det.get_cluster_info()
    print("INFO")
    print(info)
    # If running in local mode, cluster info will be None.
    if info is not None:
        latest_checkpoint = info.latest_checkpoint
        trial_id = info.trial.trial_id
    else:
        latest_checkpoint = None
        trial_id = -1

    hparams = info.trial.hparams
    data_conf = info.trial._config["data"]
    print("hparams")
    print(hparams)
    # NEW - non-hyperparameters like paths are usuallly found in the "data" part of the config file
    # cl_path = Path.joinpath(project_dir, data_conf["cl_path"])
    # cl_data_path = Path.joinpath(project_dir, data_conf["cl_data_path"])


    # NEW - replace hyperparameters values with values from config file
    base_model = hparams["base_model"]

    max_seq_length = hparams["max_seq_length"]
    train_batch_size = hparams["train_batch_size"]
    learning_rate = hparams["learning_rate"]
    warm_up_proportion = hparams["warm_up_proportion"]

    discriminate = hparams["discriminate"]
    gradual_unfreeze = hparams["gradual_unfreeze"]
    last_layer_to_freeze = hparams["last_layer_to_freeze"]
    print(hparams)

    try:
        # required if using several GPUs
        distributed = det.core.DistributedContext.from_torch_distributed()
    except:
        # in case we use a single GPU
        distributed = None

    # NEW - create a context, and pass it to the main function.
    with det.core.init(distributed=distributed) as core_context:
        main(hparams)
    DID_DET_SUCCEED = True
    # except Exception as e:
    #     if not DID_DET_SUCCEED:
    #         main(hparams)