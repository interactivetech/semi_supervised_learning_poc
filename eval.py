import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn.functional as F
import numpy as np
def predict(model,ema,data_loader,device, use_ema_model=False, return_gt=False):
        model.to(device)
        model.eval()

        if use_ema_model:
            ema.apply_shadow()

        y_true = []
        y_pred = []
        y_logits = []
        total_loss = 0.0
        total_num = 0.0
        with torch.no_grad():
            for data in tqdm(data_loader,total=len(data_loader)):
                x = data['x_lb']
                y = data['y_lb']
                num_batch = y.shape[0]
                total_num += num_batch
                if isinstance(x, dict):
                    x = {k: v.cuda(device) for k, v in x.items()}
                else:
                    x = x.cuda(device)
                y = y.cuda(device)

                logits = model(x)['logits']
                loss = F.cross_entropy(logits, y, reduction='mean')
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(torch.softmax(logits, dim=-1).cpu().numpy())
                total_loss += loss.item() * num_batch
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        total_loss /= total_num
        if use_ema_model:
            ema.restore()
        model.train()
        
        if return_gt:
            return y_pred, y_logits, y_true, total_loss
        else:
            return y_pred, y_logits, total_loss

def eval(m,ema,eval_loader,device,return_gt=True,use_ema_model=False):
    y_pred, y_logits, y_true, total_loss = predict(m,ema,eval_loader,device,return_gt=return_gt,use_ema_model=use_ema_model)
    top1 = accuracy_score(y_true, y_pred)
    balanced_top1 = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    F1 = f1_score(y_true, y_pred, average='macro')
    cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
    return top1,balanced_top1, precision, recall, F1, cf_mat