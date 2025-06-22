import torch 
import copy
from datetime import datetime
import os 
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import geoopt


def add_noise(manifold, x, noise_std):
    grad = torch.randn_like(x, device=x.device) * noise_std
    rgrad = manifold.egrad2rgrad(x, grad)
    return manifold.retr(x, rgrad)


def train_rgd(model, loss_fn, train_loader, val_loader, lrs=[1e-3, 1e-1], beta=1, betas=(0.5, 0.7), 
    wd=0, n_epochs=150, device="cpu", print_loss_step=10, patience=100, start_saving=100, save_name=None, 
    save_reps_interval=0, save_model_interval=1, noise_std=0, t_0=40, use_amsgrad=False, use_prior=False):
    
    model.to(device)
    nsample = len(train_loader.dataset)
    nsample_val = len(val_loader.dataset) if val_loader is not None else 0

    train_avg = []
    recon_avg = []
    val_avg = []
    recon_val_avg = []
    task_val_avg = []

    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0

    model_optimizer = torch.optim.Adam(model.decoder.parameters(), lr=lrs[1], weight_decay=wd, betas=betas, amsgrad=False)
    rep_optimizer = geoopt.optim.RiemannianAdam([model.z], lr=lrs[0],betas=(0.5, 0.7), amsgrad=use_amsgrad, stabilize=5)

    t_0 = t_0
    model_scheduler = CosineAnnealingWarmRestarts(model_optimizer, T_0=t_0, T_mult=1)
    rep_scheduler = CosineAnnealingWarmRestarts(rep_optimizer, T_0=t_0, T_mult=1)

    if val_loader is not None:
        valrep_optimizer = geoopt.optim.RiemannianAdam([model.z_val], lr=lrs[0], betas=(0.5, 0.7), amsgrad=use_amsgrad, stabilize=5)
        valrep_scheduler = CosineAnnealingWarmRestarts(valrep_optimizer, T_0=t_0, T_mult=1)

    if save_name is not None:
        timestamp = datetime.now().strftime('%m-%d-%H:%M')
        save_dir = os.path.join('models', 'representations', f'{save_name}_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)

    for epoch in range(n_epochs):
        train_avg.append(0)
        val_avg.append(0)
        recon_avg.append(0)
        recon_val_avg.append(0)
        task_val_avg.append(0)

        # A Gaussian prior is used for the Euclidean space. 
        # Prior weight is 0 for first 50 epochs; then linear schedule to beta at 100 epochs
        if epoch < 25: prior_weight = 0
        elif epoch < 75: prior_weight = beta * (epoch - 25) / 75
        else: prior_weight = beta

        # Train
        rep_optimizer.zero_grad()
        for batch in train_loader:
            i, data, weight = batch[0].to(device), batch[1].to(device), batch[2].to(device) 
            model_optimizer.zero_grad()
            z = model.z[i]
            if noise_std > 0: z = add_noise(model.manifold, z, noise_std)
            y = model(z).view(data.shape)
            recon_loss_x = (loss_fn(y, data, reduction='none').sum(dim=-1)).sum()
            prior_loss   = 0 if not use_prior else torch.norm(z, p=2)
            loss = recon_loss_x + prior_weight * prior_loss
            loss.backward()
            model_optimizer.step()
            train_avg[-1] += loss.item()
            recon_avg[-1] += (recon_loss_x).item()
        rep_optimizer.step()

        # Validate
        if val_loader is not None:
            if save_reps_interval > 0 and save_name is not None and (epoch % save_reps_interval == 0 or epoch <= 10):
                torch.save(model.z_val.cpu(), os.path.join(save_dir, f'z_val_epoch_{epoch}.pt'))

            valrep_optimizer.zero_grad()
            for batch in val_loader:
                i, data, weight = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                z = model.z_val[i]
                if noise_std > 0: z = add_noise(model.manifold, z, noise_std)
                y = model(z).view(data.shape)
                recon_loss_x = (loss_fn(y, data, reduction='none').sum(dim=-1)).sum()
                loss = recon_loss_x 
                loss.backward()
                val_avg[-1] += loss.item()
                recon_val_avg[-1] += (recon_loss_x).item()
            valrep_optimizer.step()

        model_scheduler.step()
        rep_scheduler.step()
        if val_loader is not None: valrep_scheduler.step()

        train_avg[-1]     /= nsample
        val_avg[-1]       /= nsample_val if nsample_val > 0 else 1
        recon_avg[-1]     /= nsample
        recon_val_avg[-1] /= nsample_val if nsample_val > 0 else 1
        task_val_avg[-1]  /= nsample_val if nsample_val > 0 else 1

        # Print epoch stats
        if epoch % print_loss_step == 0:
            print(('Epoch {:>3}  T loss: {:.4f}  V loss: {:.4f}' + \
                    '  T Recon: {:.4f}  V Recon: {:.4f}').format(
                epoch, train_avg[-1], val_avg[-1], recon_avg[-1], 
                recon_val_avg[-1]))
            
        # Early stopping
        if val_loader is not None and epoch >= start_saving:
            if val_avg[-1] <= best_val_loss:
                best_val_loss = val_avg[-1]
                if epoch >= start_saving and epoch - best_epoch >= save_model_interval:
                    # 'save_model_interval' epochs have passed since last save
                    best_model = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience and epoch >= start_saving:
                print("Early stopping")
                break

    model.load_state_dict(best_model)
    print("Best epoch:", best_epoch, "val loss:", best_val_loss)
    model.eval()

    return model