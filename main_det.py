
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
from flexmatch_det import FlexMatch


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

def main(latest_checkpoint=None,
         trial_id=None,
         hparams=None):
    '''
    '''
    # NEW - need to get GPU rank to report metrics with only one GPU and avoid duplicate reports
    rank = 0
    print("RANK: ",rank)
    print("HPARAMS: ",hparams)
    args_d = {'dataset': hparams['dataset'],
         'num_classes': hparams['num_classes'],
         'train_sampler': hparams['train_sampler'],
         'num_workers': hparams['num_workers'],
         'lb_imb_ratio': hparams['lb_imb_ratio'],
         'ulb_imb_ratio':hparams['ulb_imb_ratio'],
          'batch_size': hparams['batch_size'],
         'ulb_num_labels': hparams['ulb_num_labels'],
         'img_size': hparams['img_size'],
         'crop_ratio': hparams['crop_ratio'],
         'num_labels': hparams['num_labels'],
         'seed': hparams['seed'],
         'epoch': hparams['epoch'],
         'num_train_iter':hparams['num_train_iter'],
         'net': hparams['net'],
         'optim': hparams['optim'],
         'lr':hparams['lr'],
         'momentum':hparams['momentum'],
         'weight_decay': hparams['weight_decay'],
         'layer_decay': hparams['layer_decay'],
         'num_warmup_iter': hparams['num_warmup_iter'],
         'algorithm': hparams['algorithm'],
         'data_dir': hparams['data_dir'],
         'uratio': hparams['uratio'],
         'eval_batch_size': hparams['eval_batch_size']}
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
    
    f = FlexMatch(
             core_context,
             rank=rank,
             latest_checkpoint=latest_checkpoint,
             trial_id=trial_id,
             T=1.0, 
             p_cutoff=0.95, 
             ulb_dest_len=len(dataset_dict['train_ulb']),
             num_classes=hparams['num_classes'],
             model=m,
             ema_model=ema,
             loss_ce=loss_ce,
             scheduler=scheduler,
             optimizer=optimizer,
             device=device,
             train_lb_loader=train_lb_loader,
             train_ulb_loader=train_ulb_loader,
             eval_loader=eval_loader,
             ulb_loss_ratio=1.0,
             hard_label=True, 
             thresh_warmup=True)
    try:
        # steps, sup_loss,unsup_loss,total_loss, mask_ratio = f.fit(epochs=hparams['epoch'])
        steps, sup_loss,unsup_loss,total_loss, mask_ratio = f.fit_hyp()
    except Exception as e:
        print(e)
        pass
def driver(core_context=None,
           latest_checkpoint=None,
         trial_id=None,
         hparams=None):
    '''
    '''
    # NEW - need to get GPU rank to report metrics with only one GPU and avoid duplicate reports
    try:
        rank = torch.distributed.get_rank()
    except:
        rank = 0
    print("RANK: ",rank)
    print("HPARAMS: ",hparams)
    args_d = {'dataset': hparams['dataset'],
         'num_classes': hparams['num_classes'],
         'train_sampler': hparams['train_sampler'],
         'num_workers': hparams['num_workers'],
         'lb_imb_ratio': hparams['lb_imb_ratio'],
         'ulb_imb_ratio':hparams['ulb_imb_ratio'],
          'batch_size': hparams['batch_size'],
         'ulb_num_labels': hparams['ulb_num_labels'],
         'img_size': hparams['img_size'],
         'crop_ratio': hparams['crop_ratio'],
         'num_labels': hparams['num_labels'],
         'seed': hparams['seed'],
         'epoch': hparams['epoch'],
         'num_train_iter':hparams['num_train_iter'],
         'net': hparams['net'],
         'optim': hparams['optim'],
         'lr':hparams['lr'],
         'momentum':hparams['momentum'],
         'weight_decay': hparams['weight_decay'],
         'layer_decay': hparams['layer_decay'],
         'num_warmup_iter': hparams['num_warmup_iter'],
         'algorithm': hparams['algorithm'],
         'data_dir': hparams['data_dir'],
         'uratio': hparams['uratio'],
         'eval_batch_size': hparams['eval_batch_size']}
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
    
    f = FlexMatch(
             core_context,
             rank=rank,
             latest_checkpoint=latest_checkpoint,
             trial_id=trial_id,
             T=1.0, 
             p_cutoff=0.95, 
             ulb_dest_len=len(dataset_dict['train_ulb']),
             num_classes=hparams['num_classes'],
             model=m,
             ema_model=ema,
             loss_ce=loss_ce,
             scheduler=scheduler,
             optimizer=optimizer,
             device=device,
             train_lb_loader=train_lb_loader,
             train_ulb_loader=train_ulb_loader,
             eval_loader=eval_loader,
             ulb_loss_ratio=1.0,
             hard_label=True, 
             thresh_warmup=True)
    try:
        # steps, sup_loss,unsup_loss,total_loss, mask_ratio = f.fit(epochs=hparams['epoch'])
        steps, sup_loss,unsup_loss,total_loss, mask_ratio = f.fit_hyp()
    except Exception as e:
        print(e)
        pass
    # trainer()
def main():
    return 
def det_main(info):
    latest_checkpoint = info.latest_checkpoint
    trial_id = info.trial.trial_id
    print("info.latest_checkpoint: ",info.latest_checkpoint)
    print("trial_id: ",trial_id)
    
    hparams = info.trial.hparams
    data_conf = info.trial._config["data"]
    print("hparams")
    print(hparams)
    # NEW - non-hyperparameters like paths are usuallly found in the "data" part of the config file
    # cl_path = Path.joinpath(project_dir, data_conf["cl_path"])
    # cl_data_path = Path.joinpath(project_dir, data_conf["cl_data_path"])


    print(hparams)

    try:
        # required if using several GPUs
        distributed = det.core.DistributedContext.from_torch_distributed()
    except:
        # in case we use a single GPU
        distributed = None

    # NEW - create a context, and pass it to the main function.
    with det.core.init(distributed=distributed) as core_context:
        driver(
            core_context=core_context,
            latest_checkpoint=latest_checkpoint,
             trial_id=trial_id,
             hparams = hparams)
    DID_DET_SUCCEED = True

if __name__ == '__main__':
    # NEW - defining distributed option for det.core.init()
    DID_DET_SUCCEED=False
# try:
    info = det.get_cluster_info()
    print("INFO")
    print(info)
    if info is not None:
        det_main(info)
    else:
        main()
    # If running in local mode, cluster info will be None.
    
    # except Exception as e:
    #     if not DID_DET_SUCCEED:
    #         main(hparams)