import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
from tqdm import trange
import geoopt
import matplotlib as mpl
import torch.nn.functional as F
from scipy.stats import spearmanr


def set_plt_layout():
    mpl.style.use('seaborn-v0_8-bright')
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['text.usetex'] = False
    plt.rcParams.update({
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'axes.titlesize': 20,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.bottom': False,
        'axes.spines.left': False,
        'axes.grid': False,
        'xtick.bottom': False,
        'xtick.labelbottom': False,
        'ytick.left': False,
        'ytick.labelleft': False,
        'figure.figsize': (8, 8)
    })
    mpl.rcParams['image.cmap'] = 'magma'


def reset_plt_layout():
    plt.clf()
    plt.close('all')
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.style.use('default')


def set_all_seeds(seed): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def calculate_reconstruction_metrics(model, z, dataset):
    model.eval()
    with torch.no_grad():
        reconstructions = model(z).detach().cpu()
        target_data = dataset.data
        if isinstance(target_data, np.ndarray):
             target_data = torch.from_numpy(target_data)
        target_data = target_data.cpu() 
        mae = F.l1_loss(reconstructions, target_data)
        mse = F.mse_loss(reconstructions, target_data)
    return mae.item(), mse.item()


def calculate_reconstruction_metrics_hmtDNA(model, z, dataset, threshold=0.5):
    model.eval()
    with torch.no_grad():
        outputs = model(z)
        probs = torch.sigmoid(outputs).cpu()
        target = dataset.data
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        target = target.cpu()
        bce = F.binary_cross_entropy(probs, target, reduction='mean').item()
        preds = (probs > threshold)*1.
        tp = (preds * target).sum(dim=1)
        fp = (preds * (1 - target)).sum(dim=1)
        fn = ((1 - preds) * target).sum(dim=1)
        eps = 1e-8
        f1_per = 2 * tp / (2 * tp + fp + fn + eps)
        mean_f1 = f1_per.mean().item()
    return bce, mean_f1


def calculate_correlation_metrics_cellcycle(z, manifold, dataset, only_cc=False):
    # Filter for proliferating cells if only_cc is True
    if only_cc:
        mask = ~dataset.obs["color"].isna().values
        z_filtered = z[mask]
        theta_values = dataset.obs.cell_cycle_theta.values[mask]
        num_train_samples = len(z_filtered)
        print(f"Using {num_train_samples} proliferating cells for correlation metrics")
    else:
        z_filtered = z
        theta_values = dataset.obs.cell_cycle_theta.values
        num_train_samples = len(z)

    # Compute distance matrix between all pairs of points using the manifold's distance function
    dist_matrix = np.zeros((num_train_samples, num_train_samples))
    for i in range(num_train_samples):
        if (i + 1) % 100 == 0 or i == num_train_samples - 1: # Print progress less frequently
             print(f'Manifold distance matrix progress: {i+1}/{num_train_samples}', end='\r')
        with torch.no_grad():
            distance = manifold.dist(z_filtered[i].unsqueeze(0), z_filtered).detach().cpu().numpy().squeeze()
        dist_matrix[i] = distance

    triu_indices = np.triu_indices(num_train_samples, k=1) # upper triangular distances without redundancy
    rep_distances = dist_matrix[triu_indices]

    # Calculate cell cycle theta distances
    theta_diff = theta_values[:, None] - theta_values[None, :]
    theta_dist_matrix_abs = np.abs(theta_diff)

    # Account for circularity (theta is normalized to range [0, 1])
    theta_dist_matrix = np.minimum(theta_dist_matrix_abs, 1.0 - theta_dist_matrix_abs)
    cycle_distances = theta_dist_matrix[triu_indices]

    # Calculate correlations
    pearson_correlation = np.corrcoef(rep_distances, cycle_distances)[0, 1]
    spearman_correlation, _ = spearmanr(rep_distances, cycle_distances)

    return pearson_correlation, spearman_correlation


def calculate_correlation_metrics_hmtDNA(z, manifold, dataset, random_seed=42):
    d = dataset.data
    if len(z) > 5000:
        random.seed(random_seed) 
        indices = random.sample(range(len(z)), 5000)
        z = z[indices]
        d = dataset.data[indices]
    N = len(z)
    
    triu_idx = np.triu_indices(N, k=1)
    dist_matrix = np.zeros((N, N), dtype=float)
    for i in range(N):
        if (i + 1) % 100 == 0 or i == N - 1:
            print(f'Manifold distance progress: {i+1}/{N}', end='\r')
        with torch.no_grad():
            d_i = manifold.dist(z[i].unsqueeze(0), z).detach().cpu().numpy().squeeze()
        dist_matrix[i] = d_i
    rep_distances = dist_matrix[triu_idx]
    with torch.no_grad():
        genetic_matrix = torch.cdist(d, d, p=1).cpu().numpy()
    genetic_distances = genetic_matrix[triu_idx]
    pearson_corr = np.corrcoef(rep_distances, genetic_distances)[0, 1]
    spearman_corr, _ = spearmanr(rep_distances, genetic_distances)
    return pearson_corr, spearman_corr


def get_representations(model, loader, loss_fn, n_start_points_per_sample=100, n_epochs=1, lr=1e-3, betas=(0.5, 0.7), wd=0, device="cpu"):
    n_samples = len(loader.dataset)

    # take "n_start_points_per_sample" random samples from model.z tensor
    idx = torch.randint(0, len(model.z), (n_start_points_per_sample,))
    centers = model.z[idx]

    # repeat centers for each sample
    centers = centers.repeat(n_samples, 1, 1)
    z_init = centers.view(-1, centers.size(-1))

    # pass through model and keep the sample with minimum loss
    out = model(z_init)
    data = torch.cat([data for _, data, _ in loader], dim=0).to(device)
    loss = loss_fn(out, data.repeat_interleave(n_start_points_per_sample, dim=0)).sum(dim=-1)

    loss = loss.view(n_samples, n_start_points_per_sample, )

    # get the sample with minimum loss
    _, min_idx = torch.min(loss, dim=1)
    z_init_reshaped = z_init.view(n_samples, n_start_points_per_sample, -1)
    z_selected = z_init_reshaped[torch.arange(n_samples), min_idx]

    # Make z_selected a leaf variable with requires_grad=True
    z = z_selected.clone().detach().to(device)
    z = model.manifold.projx(z)
    z = geoopt.ManifoldParameter(z, manifold=model.manifold, requires_grad=True)
    optimizer = geoopt.optim.RiemannianAdam([z], lr=lr, weight_decay=wd, betas=betas, stabilize=10)
    model.to(device)

    for epoch in trange(n_epochs, desc="Optimizing representations"):
        optimizer.zero_grad()
        for batch in loader:
            i, data, _ = batch
            i, data = i.to(device), data.to(device)
            z_subset = z[i]
            y = model(z_subset)
            loss = loss_fn(y, data).sum()
            loss.backward()
        optimizer.step()
    return z
