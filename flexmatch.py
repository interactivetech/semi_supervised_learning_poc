import torch
from copy import deepcopy
from collections import Counter

from loss import ce_loss, consistency_loss, smooth_targets
from ema import EMA,EMADriver, set_ema_model
from tqdm import tqdm

class PseudoLabelingHook:
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def gen_ulb_targets(self, 
                        logits, 
                        use_hard_label=True, 
                        T=1.0,
                        softmax=True, # whether to compute softmax for logits, input must be logits
                        label_smoothing=0.0):
                        
        logits = logits.detach()
        if use_hard_label:
            # return hard label directly
            pseudo_label = torch.argmax(logits, dim=-1)
            if label_smoothing:
                pseudo_label = smooth_targets(logits, pseudo_label, label_smoothing)
            return pseudo_label
        
        # return soft label
        if softmax:
            pseudo_label = torch.softmax(logits / T, dim=-1)
        else:
            # inputs logits converted to probabilities already
            pseudo_label = logits
        return pseudo_label

class FlexMatchThresholdingHook:
    """
    Adaptive Thresholding in FlexMatch
    """
    def __init__(self, 
                 ulb_dest_len, 
                 num_classes,
                 T,
                 p_cutoff,
                 thresh_warmup=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ulb_dest_len = ulb_dest_len
        # print("self.ulb_dest_len: ",self.ulb_dest_len) 
        self.num_classes = num_classes
        self.thresh_warmup = thresh_warmup
        self.T = T
        self.p_cutoff = p_cutoff
        self.thresh_warmup = thresh_warmup
        self.selected_label = torch.ones((self.ulb_dest_len,), dtype=torch.long, ) * -1
        # print("self.selected_label: ",self.selected_label) 

        self.classwise_acc = torch.zeros((self.num_classes,))

    @torch.no_grad()
    def update(self, *args, **kwargs):
        pseudo_counter = Counter(self.selected_label.tolist())
        '''
        Example of pseudo_counter: Counter({-1: 141, 0: 3, 2: 3, 9: 2, 4: 1})
        '''
        # print("pseudo_counter: ",pseudo_counter)
        if max(pseudo_counter.values()) < self.ulb_dest_len:  # not all(5w) -1
            if self.thresh_warmup:
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
                    # print("pseudo_counter[i] / max(pseudo_counter.values()): ", 
                    #       pseudo_counter[i],
                    #       max(pseudo_counter.values()),
                    #       pseudo_counter[i] / max(pseudo_counter.values()))
                    # print("t-i, self.classwise_acc: ",i,self.classwise_acc)
            else:
                wo_negative_one = deepcopy(pseudo_counter)
                if -1 in wo_negative_one.keys():
                    wo_negative_one.pop(-1)
                for i in range(self.num_classes):
                    self.classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())
                    # print("t-i, self.classwise_acc: ",i,self.classwise_acc)
                    # print("pseudo_counter[i] / max(pseudo_counter.values()): ", 
                    #       pseudo_counter[i],
                    #       max(pseudo_counter.values()),
                    #       pseudo_counter[i] / max(pseudo_counter.values()))

    @torch.no_grad()
    def masking(self, logits_x_ulb, idx_ulb, softmax_x_ulb=True, *args, **kwargs):
        if not self.selected_label.is_cuda:
            self.selected_label = self.selected_label.to(logits_x_ulb.device)
        if not self.classwise_acc.is_cuda:
            self.classwise_acc = self.classwise_acc.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1)
        # print("max_probs: ", max_probs.shape, max_probs)
        # print("max_idx: ", max_idx.shape, max_idx)
        # mask = max_probs.ge(p_cutoff * (class_acc[max_idx] + 1.) / 2).float()  # linear
        # mask = max_probs.ge(p_cutoff * (1 / (2. - class_acc[max_idx]))).float()  # low_limit
        mask = max_probs.ge(self.p_cutoff * (self.classwise_acc[max_idx] / (2. - self.classwise_acc[max_idx])))  # convex
        # print("mask: ", mask.shape, mask)
        # mask = max_probs.ge(p_cutoff * (torch.log(class_acc[max_idx] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave
        select = max_probs.ge(self.p_cutoff)
        mask = mask.to(max_probs.dtype)

        # update
        if idx_ulb[select == 1].nelement() != 0:
            self.selected_label[idx_ulb[select == 1]] = max_idx[select == 1]
        self.update()

        return mask
        
        

class FlexMatch:
    """
        FlexMatch algorithm (https://arxiv.org/abs/2110.08263).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - ulb_dest_len (`int`):
                Length of unlabeled data
            - thresh_warmup (`bool`, *optional*, default to `True`):
                If True, warmup the confidence threshold, so that at the beginning of the training, all estimated
                learning effects gradually rise from 0 until the number of unused unlabeled data is no longer
                predominant

        """
    def __init__(self, 
             T=None, 
             p_cutoff=None, 
             ulb_dest_len=None,
             num_classes=None,
             model=None,
             ema_model=None,
             loss_ce=None,
             scheduler=None,
             optimizer=None,
             device=None,
             train_lb_loader=None,
             train_ulb_loader=None,
             ulb_loss_ratio=1.0,
             hard_label=True, 
             thresh_warmup=True):
        super().__init__()
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.ulb_dest_len = ulb_dest_len
        self.lambda_u =ulb_loss_ratio
        self.num_classes = num_classes
        self.thresh_warmup = thresh_warmup
        self.model=model
        self.ema_model = ema_model
        self.loss_ce = loss_ce
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.use_cat = True
        self.train_lb_loader = train_lb_loader
        self.train_ulb_loader = train_ulb_loader
        self.threshold = FlexMatchThresholdingHook(ulb_dest_len=self.ulb_dest_len, 
                                                   num_classes=self.num_classes, 
                                                   T=self.T,
                                                   p_cutoff=self.p_cutoff,
                                                   thresh_warmup=self.thresh_warmup)
        self.pseudolabel = PseudoLabelingHook()
        self.device = device
        self.init_ema()
        # super().set_hooks()
        
        
    def init_ema(self):
        '''
        '''
        self.model.to(self.device)
        self.ema_model.to(self.device)
        self.ema_model = set_ema_model(self.ema_model, self.model)
        self.emaA = EMADriver(model=self.model,ema_model=self.ema_model,ema_m=0.999)
        self.emaA.before_run()
        
    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        # with self.amp_cm():
        #     if self.use_cat:
        #         inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        #         outputs = self.model(inputs)
        #         logits_x_lb = outputs['logits'][:num_lb]
        #         logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
        #     else:
        #         outs_x_lb = self.model(x_lb) 
        #         logits_x_lb = outs_x_lb['logits']
        #         outs_x_ulb_s = self.model(x_ulb_s)
        #         logits_x_ulb_s = outs_x_ulb_s['logits']
        #         with torch.no_grad():
        #             outs_x_ulb_w = self.model(x_ulb_w)
        #             logits_x_ulb_w = outs_x_ulb_w['logits']
        self.optimizer.zero_grad()

        if self.use_cat:
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
            outputs = self.model(inputs)
            logits_x_lb = outputs['logits'][:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
        else:
            outs_x_lb = self.model(x_lb) 
            logits_x_lb = outs_x_lb['logits']
            outs_x_ulb_s = self.model(x_ulb_s)
            logits_x_ulb_s = outs_x_ulb_s['logits']
            with torch.no_grad():
                outs_x_ulb_w = self.model(x_ulb_w)
                logits_x_ulb_w = outs_x_ulb_w['logits']

        sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

        # compute mask
        mask = self.threshold.masking(logits_x_ulb=logits_x_ulb_w, idx_ulb=idx_ulb)

        # generate unlabeled targets using pseudo label hook
        pseudo_label = self.pseudolabel.gen_ulb_targets(logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)

        unsup_loss = consistency_loss(logits_x_ulb_s,
                                      pseudo_label,
                                      'ce',
                                      mask=mask)

        total_loss = sup_loss + self.lambda_u * unsup_loss
        
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # parameter updates
        # self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = mask.float().mean().item()
        return tb_dict
    def fit(self,epochs = 10):
        losses = []
        sup_loss = []
        unsup_loss = []
        total_loss = []
        mask_ratio = []
        steps = []
        total_steps = 0
        for e in tqdm(range(epochs)):
            for ind,(data_lb, data_ulb) in enumerate(zip(self.train_lb_loader, self.train_ulb_loader)):
                # print(data_lb.keys())
                idx_lb = data_lb['idx_lb'].to(self.device)
                x_lb = data_lb['x_lb'].to(self.device)
                y_lb = data_lb['y_lb'].to(self.device)
                idx_ulb = data_ulb['idx_ulb'].to(self.device)
                x_ulb_w = data_ulb['x_ulb_w'].to(self.device)
                x_ulb_s = data_ulb['x_ulb_s'].to(self.device)
                # print("idx_lb: ",idx_lb)
                # print("idx_ulb: ",idx_ulb)
                loss = self.train_step( x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s)
                
                if total_steps%10==0:
                    with torch.no_grad():
                        sup_loss.append(loss['train/sup_loss'])
                        unsup_loss.append(loss['train/unsup_loss'])
                        total_loss.append(loss['train/total_loss'])
                        mask_ratio.append(loss['train/mask_ratio'])
                        steps.append(total_steps)
                    # print(loss.item())
                total_steps+=1
                self.emaA.after_train_step()
        return steps, sup_loss,unsup_loss,total_loss, mask_ratio
        
        
