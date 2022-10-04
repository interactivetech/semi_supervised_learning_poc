from tqdm import tqdm
import torch
def trainer(train_lb_loader,m,ema,loss_ce,scheduler,optimizer, train_ulb_loader,device,epochs=10):
    losses = []
    steps = []
    total_steps = 0
    for _ in tqdm(range(epochs)):
        for ind,(data_lb, data_ulb) in enumerate(zip(train_lb_loader, train_ulb_loader)):
            x = data_lb['x_lb'].to(device)
            y = data_lb['y_lb'].to(device)
            optimizer.zero_grad()

            out = m(x)
            out = out['logits']
            loss = loss_ce(out,y)

            loss.backward()
            optimizer.step()
            scheduler.step()
            if total_steps%10==0:
                with torch.no_grad():
                    losses.append(loss.item())
                    steps.append(total_steps)
                    # print(loss.item())
            total_steps+=1
            ema.after_train_step()
    return steps,losses