import numpy as np
from tqdm import tqdm
from collections import defaultdict
import argparse
import json
import random
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as FT

### Internals
from manifmetric.manifmetric import ManifoldMetric, VGG16, Logger
from manifmetric.dataset import CelebADataset, CIFAR10Dataset, block_draw

METRIC_STR = {
    'precision': 'Precision',
    'recall': 'Recall',
    'comp_precision': 'cPrecision',
    'coverage': 'cRecall',
    'sym_precision': 'symPrecision',
    'sym_recall': 'symRecall'
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

### Toy Experiments
def hypercube_exp(save_dir,
        ref_r=1, lower=1., upper=1., step=0.1, knn=5, dims=32, 
        data_size=10000, ref_size=None, gen_size=None, runs=1, device='cpu', seed=1000, dtype=torch.float32, **kwargs):
    '''
    @ref_r: half-edge length of the reference hypercube.
    @lower: ref_r-lower determines the inclusive lower bound half-edge length of generated hypercube.
    @upper: ref_r+upper determines the inclusive upper bound half-edge length of generated hypercube.
    @step: step size on the half-edge length of the generated hypercube in [ref_r-lower, ref_r+upper].
    @knn: k of k-nearest neighbors in computing metrics.
    @dims: int or list of dimensions to consider.
    @save_dir: directory to save results.
    @data_size: number of samples from ref and gen supports (used as default value when ref_size=None and/or gen_size=None).
    @runs: number of runs.
    @device: device to compute the metrics.
    @seed: seed+run_id is used for generating ref and gen samples.
    @dtype: dtype of samples and metrics computations.
    '''
    ### Setup
    ref_size = data_size if ref_size is None else ref_size
    gen_size = data_size if gen_size is None else gen_size

    print('>>> Hypercube experiment with the following arguments:')
    for arg_name, arg_val in locals().items():
        if isinstance(arg_val, (int, float, str, dict, list, tuple)):
            print(f'{arg_name}: {arg_val}')
        else:
            print(f'{arg_name}: {type(arg_val).__name__}')

    rng = torch.Generator(device=device)
    dims = [dims] if isinstance(dims, int) else dims
    scales = np.arange(-lower, upper+step, step) + ref_r
    metrics = defaultdict(lambda: {dim: np.zeros((len(scales), runs)) for dim in dims})
    
    ### Run
    pbar = tqdm(total=runs*len(dims)*len(scales))
    for run in range(runs):
        rng.manual_seed(seed+run)
        for dim in dims:
            manif_metric = ManifoldMetric()

            ### Reference data
            ref_data = 2 * ref_r * (torch.rand((ref_size, dim), generator=rng, dtype=dtype, device=device) - 0.5)
            axes = torch.randint(dim, (ref_size,), generator=rng, dtype=torch.int64, device=device)
            ref_data[torch.arange(ref_size).to(device), axes] =  ref_r * torch.as_tensor([1., -1.], dtype=dtype, device=device)[
                torch.randint(0, 2, (ref_size,), generator=rng, dtype=torch.int64, device=device)]
            manif_metric.compute_ref_stats(ref_data, k=knn)

            for si, scale in enumerate(scales):
                pbar.update(1)
                pbar.set_description(f'Metric Run={run} Dim={dim} Scale={scale:.2f}')

                ### Model data
                gen_data = 2 * scale * (torch.rand((gen_size, dim), generator=rng, dtype=dtype, device=device) - 0.5)
                axes = torch.randint(dim, (gen_size,), generator=rng, dtype=torch.int64, device=device)
                gen_data[torch.arange(gen_size).to(device), axes] =  scale * torch.as_tensor([1., -1.], dtype=dtype, device=device)[
                    torch.randint(0, 2, (gen_size,), generator=rng, dtype=torch.int64, device=device)]
                manif_metric.compute_gen_stats(gen_data, k=knn)

                ### Compute metrics
                precs = manif_metric.sym_precision()
                recalls = manif_metric.sym_recall()
                
                metrics['precision'][dim][si, run] = precs.precision
                metrics['comp_precision'][dim][si, run] = precs.comp_precision
                metrics['sym_precision'][dim][si, run] = precs.sym_precision
                
                metrics['recall'][dim][si, run] = recalls.recall
                metrics['coverage'][dim][si, run] = recalls.coverage
                metrics['sym_recall'][dim][si, run] = recalls.sym_recall
                
                ### Conserve memory
                del gen_data
                manif_metric.gen_stats = None
    pbar.close()

    ### Save and plot
    title='Hypercube'
    xlabel='Half Edge-length of Generated Hypercube'
    with open(os.path.join(save_dir, f'{title.lower().replace(" ", "_")}_metrics.json'), 'w+') as fs:
        json.dump(dict(scales=scales, metrics=dict(metrics)), fs, cls=NumpyEncoder, indent=2)
    
    colors = matplotlib.colormaps['plasma'](np.linspace(0., .8, len(dims)))
    markers = ['.', 's', 'v', '^', 'd', 'x', '1', '2', '3', '4'][:len(dims)]
    for name, metric in metrics.items():
        fig = plt.figure(figsize=(8,6), constrained_layout=True)
        ax = fig.add_subplot()
        xticks = np.arange(len(scales))
        for di, dim in enumerate(dims):
            mean_metric = np.mean(metric[dim], axis=1)
            std_metric = np.std(metric[dim], axis=1)
            ax.plot(xticks, mean_metric,
                label=f'd={dim}', color=colors[di], marker=markers[di], alpha=0.8)
            ax.fill_between(xticks, mean_metric-std_metric, mean_metric+std_metric,
                color=colors[di], alpha=0.2)
        ax.legend(loc=0)
        ax.grid(True, which='both', linestyle='dotted')
        
        ax.set_xticks(xticks)
        xticks_labels = [' '] * len(xticks)
        xticks_labels[0] = f'{scales[0]:0.1f}'
        xticks_labels[len(xticks)//2] = f'{scales[len(xticks)//2]:0.1f}'
        xticks_labels[-1] = f'{scales[-1]:0.1f}'
        ax.set_xticklabels(xticks_labels)

        ax.set_ylim(bottom=0, top=1.2)
        ### Hide top yticklabel
        yticks = np.arange(0, 1.4, 0.2)
        yticklabels = [f'{ytick:0.1f}' for ytick in yticks]
        yticklabels[-1] = ''
        ax.set_yticks(yticks, yticklabels)

        ### Setup titles
        ax.set_title(f'{title} {METRIC_STR[name]}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(METRIC_STR[name])

        fig.savefig(os.path.join(save_dir, f'{title.lower().replace(" ", "_")}_{name}.pdf'))

def hypersphere_exp(save_dir, 
        ref_r=1, lower=1., upper=1., step=0.1, knn=5, dims=32, 
        data_size=10000, ref_size=None, gen_size=None, runs=1, device='cpu', seed=1000, dtype=torch.float32, **kwargs):
    '''
    @ref_r: radius the reference hypersphere.
    @lower: ref_r-lower determines the inclusive lower bound radius generated hypersphere.
    @upper: ref_r+upper determines the inclusive upper bound radius generated hypersphere.
    @step: step size on the radius the generated hypersphere in [ref_r-lower, ref_r+upper].
    @knn: k of k-nearest neighbors in computing metrics.
    @dims: int or list of dimensions to consider.
    @save_dir: directory to save results.
    @data_size: number of samples from ref and gen supports (used as default value when ref_size=None and/or gen_size=None).
    @runs: number of runs.
    @device: device to compute the metrics.
    @seed: seed+run_id is used for generating ref and gen samples.
    @dtype: dtype of samples and metrics computations.
    '''
    ### Setup
    ref_size = data_size if ref_size is None else ref_size
    gen_size = data_size if gen_size is None else gen_size

    print('>>> Hypersphere experiment with the following arguments:')
    for arg_name, arg_val in locals().items():
        if isinstance(arg_val, (int, float, str, dict, list, tuple)):
            print(f'{arg_name}: {arg_val}')
        else:
            print(f'{arg_name}: {type(arg_val).__name__}')
    
    rng = torch.Generator(device=device)
    dims = [dims] if isinstance(dims, int) else dims
    scales = np.arange(-lower, upper+step, step) + ref_r
    metrics = defaultdict(lambda: {dim: np.zeros((len(scales), runs)) for dim in dims})
    
    ### Run
    pbar = tqdm(total=runs*len(dims)*len(scales))
    for run in range(runs):
        rng.manual_seed(seed+run)
        for dim in dims:
            manif_metric = ManifoldMetric()

            ### Reference data
            ref_data = torch.randn((ref_size, dim), generator=rng, dtype=dtype, device=device)
            ref_data = ref_r * ref_data / ref_data.pow(2).sum(1, keepdim=True).sqrt()
            manif_metric.compute_ref_stats(ref_data, k=knn)

            for si, scale in enumerate(scales):
                pbar.update(1)
                pbar.set_description(f'Metric Run={run} Dim={dim} Scale={scale:.2f}')

                ### Model data
                gen_data = torch.randn((gen_size, dim), generator=rng, dtype=dtype, device=device)
                gen_data = scale * gen_data / gen_data.pow(2).sum(1, keepdim=True).sqrt()
                manif_metric.compute_gen_stats(gen_data, k=knn)

                ### Compute metrics
                precs = manif_metric.sym_precision()
                recalls = manif_metric.sym_recall()
                
                metrics['precision'][dim][si, run] = precs.precision
                metrics['comp_precision'][dim][si, run] = precs.comp_precision
                metrics['sym_precision'][dim][si, run] = precs.sym_precision
                
                metrics['recall'][dim][si, run] = recalls.recall
                metrics['coverage'][dim][si, run] = recalls.coverage
                metrics['sym_recall'][dim][si, run] = recalls.sym_recall

                ### Conserve memory
                del gen_data
                manif_metric.gen_stats = None
    pbar.close()

    ### Save and plot
    title='Hypersphere'
    xlabel='Radius of Generated Hypersphere'
    with open(os.path.join(save_dir, f'{title.lower().replace(" ", "_")}_metrics.json'), 'w+') as fs:
        json.dump(dict(scales=scales, metrics=dict(metrics)), fs, cls=NumpyEncoder, indent=2)
    
    colors = matplotlib.colormaps['plasma'](np.linspace(0., .8, len(dims)))
    markers = ['.', 's', 'v', '^', 'd', 'x', '1', '2', '3', '4'][:len(dims)]
    for name, metric in metrics.items():
        fig = plt.figure(figsize=(8,6), constrained_layout=True)
        ax = fig.add_subplot()
        xticks = np.arange(len(scales))
        for di, dim in enumerate(dims):
            mean_metric = np.mean(metric[dim], axis=1)
            std_metric = np.std(metric[dim], axis=1)
            ax.plot(xticks, mean_metric,
                label=f'd={dim}', color=colors[di], marker=markers[di], alpha=0.8)
            ax.fill_between(xticks, mean_metric-std_metric, mean_metric+std_metric,
                color=colors[di], alpha=0.2)
        ax.legend(loc=0)
        ax.grid(True, which='both', linestyle='dotted')
        
        ax.set_xticks(xticks)
        xticks_labels = [' '] * len(xticks)
        xticks_labels[0] = f'{scales[0]:0.1f}'
        xticks_labels[len(xticks)//2] = f'{scales[len(xticks)//2]:0.1f}'
        xticks_labels[-1] = f'{scales[-1]:0.1f}'
        ax.set_xticklabels(xticks_labels)

        ax.set_ylim(bottom=0, top=1.2)
        ### Hide top yticklabel
        yticks = np.arange(0, 1.4, 0.2)
        yticklabels = [f'{ytick:0.1f}' for ytick in yticks]
        yticklabels[-1] = ''
        ax.set_yticks(yticks, yticklabels)

        ### Setup titles
        ax.set_title(f'{title} {METRIC_STR[name]}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(METRIC_STR[name])

        fig.savefig(os.path.join(save_dir, f'{title.lower().replace(" ", "_")}_{name}.pdf'))

def normal_hypersphere_exp(save_dir, 
        lower=1., upper=1., step=0.1, knn=5, dims=32,
        data_size=10000, ref_size=None, gen_size=None, runs=1, device='cpu', seed=1000, dtype=torch.float32, **kwargs):
    '''
    @lower: ref_r-lower determines the inclusive lower bound radius generated hypersphere (ref_r=sqrt(dim)).
    @upper: ref_r+upper determines the inclusive upper bound radius generated hypersphere (ref_r=sqrt(dim)).
    @step: step size on the radius the generated hypersphere in [ref_r-lower, ref_r+upper].
    @knn: k of k-nearest neighbors in computing metrics.
    @dims: int or list of dimensions to consider.
    @save_dir: directory to save results.
    @data_size: number of samples from ref and gen supports (used as default value when ref_size=None and/or gen_size=None).
    @runs: number of runs.
    @device: device to compute the metrics.
    @seed: seed+run_id is used for generating ref and gen samples.
    @dtype: dtype of samples and metrics computations.
    '''
    ### Setup
    ref_size = data_size if ref_size is None else ref_size
    gen_size = data_size if gen_size is None else gen_size

    print('>>> Normal hypersphere experiment with the following arguments:')
    for arg_name, arg_val in locals().items():
        if isinstance(arg_val, (int, float, str, dict, list, tuple)):
            print(f'{arg_name}: {arg_val}')
        else:
            print(f'{arg_name}: {type(arg_val).__name__}')

    rng = torch.Generator(device=device)
    dims = [dims] if isinstance(dims, int) else dims
    base_scales = np.arange(-lower, upper+step, step)
    metrics = defaultdict(lambda: {dim: np.zeros((len(base_scales), runs)) for dim in dims})
    
    ### Run
    pbar = tqdm(total=runs*len(dims)*len(base_scales))
    for run in range(runs):
        rng.manual_seed(seed+run)
        for dim in dims:
            manif_metric = ManifoldMetric()
            ref_r = np.sqrt(dim)
            scales = base_scales + ref_r

            ### Reference data
            ref_data = torch.randn((ref_size, dim), generator=rng, dtype=dtype, device=device)
            ref_data = ref_r * ref_data / ref_data.pow(2).sum(1, keepdim=True).sqrt()
            manif_metric.compute_ref_stats(ref_data, k=knn)

            for si, scale in enumerate(scales):
                pbar.update(1)
                pbar.set_description(f'Metric Run={run} Dim={dim} Scale={scale:.2f}')

                ### Model data
                gen_data = torch.randn((gen_size, dim), generator=rng, dtype=dtype, device=device)
                gen_data = scale * gen_data / gen_data.pow(2).sum(1, keepdim=True).sqrt()
                manif_metric.compute_gen_stats(gen_data, k=knn)

                ### Compute metrics
                precs = manif_metric.sym_precision()
                recalls = manif_metric.sym_recall()
                
                metrics['precision'][dim][si, run] = precs.precision
                metrics['comp_precision'][dim][si, run] = precs.comp_precision
                metrics['sym_precision'][dim][si, run] = precs.sym_precision
                
                metrics['recall'][dim][si, run] = recalls.recall
                metrics['coverage'][dim][si, run] = recalls.coverage
                metrics['sym_recall'][dim][si, run] = recalls.sym_recall

                ### Conserve memory
                del gen_data
                manif_metric.gen_stats = None
    pbar.close()

    ### Save and plot
    title='Standard Normal Hypersphere'
    xlabel='Radius of Generated Hypersphere'
    with open(os.path.join(save_dir, f'{title.lower().replace(" ", "_")}_metrics.json'), 'w+') as fs:
        json.dump(dict(scales=scales, metrics=dict(metrics)), fs, cls=NumpyEncoder, indent=2)

    colors = matplotlib.colormaps['plasma'](np.linspace(0., .8, len(dims)))
    markers = ['.', 's', 'v', '^', 'd', 'x', '1', '2', '3', '4'][:len(dims)]
    for name, metric in metrics.items():
        fig = plt.figure(figsize=(8,6), constrained_layout=True)
        ax = fig.add_subplot()
        xticks = np.arange(len(scales))
        for di, dim in enumerate(dims):
            mean_metric = np.mean(metric[dim], axis=1)
            std_metric = np.std(metric[dim], axis=1)
            ax.plot(xticks, mean_metric,
                label=f'd={dim}', color=colors[di], marker=markers[di], alpha=.8)
            ax.fill_between(xticks, mean_metric-std_metric, mean_metric+std_metric,
                color=colors[di], alpha=0.2)
        ax.legend(loc=0 if 'sym' not in name else 8)
        ax.grid(True, which='both', linestyle='dotted')

        ax.set_xticks(xticks)
        xticks_labels = [' '] * len(xticks)
        xticks_labels[0] = f'$\\sqrt{{d}}$-{int(abs(lower))}'
        xticks_labels[len(xticks)//2] = f'$\\sqrt{{d}}$'
        xticks_labels[-1] = f'$\\sqrt{{d}}$+{int(abs(upper))}'
        ax.set_xticklabels(xticks_labels)
        
        ax.set_ylim(bottom=0, top=1.2)
        ### Hide top yticklabel
        yticks = np.arange(0, 1.4, 0.2)
        yticklabels = [f'{ytick:0.1f}' for ytick in yticks]
        yticklabels[-1] = ''
        ax.set_yticks(yticks, yticklabels)

        ### Setup titles
        ax.set_title(f'{title} {METRIC_STR[name]}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(METRIC_STR[name])

        fig.savefig(os.path.join(save_dir, f'{title.lower().replace(" ", "_")}_{name}.pdf'))

### Real Data Experiments
def feature_scaled_exp(save_dir, model, dataset, 
    lower=0.5, upper=1.5, step=0.1, knn=5,
    data_size=10000, ref_size=None, gen_size=None, runs=1, device='cpu', seed=1000, dtype=torch.float32, batch_size=32, num_workers=2, **kwargs):
    '''
    @model: name of model or the model as a function to be used by ManifoldMetric.
    @dataset: dataset to use, must have lists of dataset indices dataset.train_set and dataset.test_set.
    @lower: lower determines the inclusive lower bound of contrast scale.
    @upper: upper determines the inclusive upper bound of contrast scale.
    @step: step size on the contrast scale.
    @knn: k of k-nearest neighbors in computing metrics.
    @save_dir: directory to save results.
    @data_size: number of samples from ref and gen supports (used as default value when ref_size=None and/or gen_size=None).
    @runs: number of runs.
    @device: device to compute the metrics.
    @seed: seed+run_id is used for generating ref and gen samples.
    @dtype: dtype of samples and metrics computations.
    @batch_size: batch_size used for feature extraction.
    @num_workers: num_workers used for loading images from dataset.
    '''
    ### Setup
    ref_size = data_size if ref_size is None else ref_size
    gen_size = data_size if gen_size is None else gen_size

    print('>>> Feature scaled experiment with the following arguments:')
    for arg_name, arg_val in locals().items():
        if isinstance(arg_val, (int, float, str, dict, list, tuple)):
            print(f'{arg_name}: {arg_val}')
        else:
            print(f'{arg_name}: {type(arg_val).__name__}')

    assert (ref_size <= len(dataset.test_set))
    assert (gen_size <= len(dataset.train_set))
    scales = np.arange(lower, upper+step, step)
    metrics = defaultdict(lambda: np.zeros((len(scales), runs), dtype=float))
    manif_metric = ManifoldMetric(model=model)

    def transformer(data):
        imgs = data[0] if isinstance(data, (tuple, list)) else data
        imgs = imgs.permute(0, 3, 1, 2) # BHWC to BCHW
        if imgs.shape[1] == 1:
            imgs = imgs.repeat([1, 3, 1, 1])
        return imgs
    
    ### Run
    pbar = tqdm(total=runs*len(scales))
    for run in range(runs):
        rng = np.random.default_rng(seed+run)

        ### Reference data
        test_loader = DataLoader(dataset=dataset, sampler=rng.permutation(dataset.test_set)[:ref_size], batch_size=batch_size, num_workers=num_workers)
        ref_data = manif_metric.extract_features(test_loader, transform=transformer, device=device, dtype=dtype)
        manif_metric.compute_ref_stats(ref_data, k=knn)

        ### Train features
        train_loader = DataLoader(dataset=dataset, sampler=rng.permutation(dataset.train_set)[:gen_size], batch_size=batch_size, num_workers=num_workers)
        train_feats = manif_metric.extract_features(train_loader, transform=transformer, device=device, dtype=dtype)
        train_feats_mean = train_feats.mean(dim=0, keepdim=True)
        for si, scale in enumerate(scales):
            pbar.update(1)
            pbar.set_description(f'Metric Run={run} Scale={scale:.2f}')
            
            ### Generated data
            gen_data = scale * (train_feats - train_feats_mean) + train_feats_mean
            manif_metric.compute_gen_stats(gen_data, k=knn)

            ### Compute metrics
            precs = manif_metric.sym_precision()
            recalls = manif_metric.sym_recall()
            
            metrics['precision'][si, run] = precs.precision
            metrics['comp_precision'][si, run] = precs.comp_precision
            metrics['sym_precision'][si, run] = precs.sym_precision
            
            metrics['recall'][si, run] = recalls.recall
            metrics['coverage'][si, run] = recalls.coverage
            metrics['sym_recall'][si, run] = recalls.sym_recall

            ### Conserve memory
            del gen_data
            manif_metric.gen_stats = None
    pbar.close()

    ### Save and plot
    title=f'{type(dataset).__name__}'
    xlabel='Scale of Feature Expansion'
    with open(os.path.join(save_dir, f'{title.lower().replace(" ", "_")}_metrics.json'), 'w+') as fs:
        json.dump(dict(scales=scales, metrics=dict(metrics)), fs, cls=NumpyEncoder, indent=2)

    colors = matplotlib.colormaps['plasma'](np.linspace(0., .8, 3))
    markers = ['.', 's', 'v', '^', 'd', 'x', '1', '2', '3', '4']
    linestyles = ['--', '--', None]
    
    metric_groups = {
        'precision': ['precision', 'comp_precision', 'sym_precision'],
        'recall': ['recall', 'coverage', 'sym_recall']}
    
    mean_metrics = dict()
    std_metrics = dict()
    for group_name, metric_group in metric_groups.items():
        fig = plt.figure(figsize=(8,6), constrained_layout=True)
        ax = fig.add_subplot()
        xticks = np.arange(len(scales))
        for ni, name in enumerate(metric_group):
            metric = metrics[name]
            mean_metric = np.mean(metric, axis=1)
            std_metric = np.std(metric, axis=1)
            ax.plot(xticks, mean_metric,
                label=METRIC_STR[name], color=colors[ni], marker=markers[ni], alpha=0.8, linestyle=linestyles[ni])
            ax.fill_between(xticks, mean_metric-std_metric, mean_metric+std_metric,
                color=colors[ni], alpha=0.2)
            mean_metrics[name] = mean_metric
            std_metrics[name] = std_metric
        ax.legend(loc=8)
        ax.grid(True, which='both', linestyle='dotted')
        
        ax.set_xticks(xticks)
        xticks_labels = [' '] * len(xticks)
        xticks_labels[0] = f'{scales[0]:0.1f}'
        xticks_labels[len(xticks)//2] = f'{scales[len(xticks)//2]:0.1f}'
        xticks_labels[-1] = f'{scales[-1]:0.1f}'
        ax.set_xticklabels(xticks_labels)

        ax.set_ylim(bottom=0, top=1.2)
        ### Hide top yticklabel
        yticks = np.arange(0, 1.4, 0.2)
        yticklabels = [f'{ytick:0.1f}' for ytick in yticks]
        yticklabels[-1] = ''
        ax.set_yticks(yticks, yticklabels)

        ### Setup titles
        ax.set_title(f'{title} {METRIC_STR[group_name]}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(METRIC_STR[group_name])

        fig.savefig(os.path.join(save_dir, f'{title.lower().replace(" ", "_")}_{group_name}.pdf'))

def contrast_exp(save_dir, model, dataset, 
    lower=0.0, upper=2.0, step=0.2, knn=5,
    data_size=10000, ref_size=None, gen_size=None, runs=1, device='cpu', seed=1000, dtype=torch.float32, batch_size=32, num_workers=2, **kwargs):
    '''
    @model: name of model or the model as a function to be used by ManifoldMetric.
    @dataset: dataset to use, must have lists of dataset indices dataset.train_set and dataset.test_set.
    @lower: lower determines the inclusive lower bound of contrast scale.
    @upper: upper determines the inclusive upper bound of contrast scale.
    @step: step size on the contrast scale.
    @knn: k of k-nearest neighbors in computing metrics.
    @save_dir: directory to save results.
    @data_size: number of samples from ref and gen supports (used as default value when ref_size=None and/or gen_size=None).
    @runs: number of runs.
    @device: device to compute the metrics.
    @seed: seed+run_id is used for generating ref and gen samples.
    @dtype: dtype of samples and metrics computations.
    @batch_size: batch_size used for feature extraction.
    @num_workers: num_workers used for loading images from dataset.
    '''
    ### Setup
    ref_size = data_size if ref_size is None else ref_size
    gen_size = data_size if gen_size is None else gen_size

    print('>>> Contrast experiment with the following arguments:')
    for arg_name, arg_val in locals().items():
        if isinstance(arg_val, (int, float, str, dict, list, tuple)):
            print(f'{arg_name}: {arg_val}')
        else:
            print(f'{arg_name}: {type(arg_val).__name__}')

    assert (ref_size <= len(dataset.test_set))
    assert (gen_size <= len(dataset.train_set))
    scales = np.arange(lower, upper+step, step)
    metrics = defaultdict(lambda: np.zeros((len(scales), runs), dtype=float))
    manif_metric = ManifoldMetric(model=model)
    
    def transformer(scale=1.):
        def transform_fn(data):
            imgs = data[0] if isinstance(data, (tuple, list)) else data
            imgs = imgs.permute(0, 3, 1, 2) # BHWC to BCHW
            if imgs.shape[1] == 1:
                imgs = imgs.repeat([1, 3, 1, 1])
            return FT.adjust_contrast(imgs, scale) if scale != 1 else imgs
        return transform_fn
    
    ### Run
    pbar = tqdm(total=runs*len(scales))
    for run in range(runs):
        collect = list()
        rng = np.random.default_rng(seed+run)

        ### Reference data
        test_loader = DataLoader(dataset=dataset, sampler=rng.permutation(dataset.test_set)[:ref_size], batch_size=batch_size, num_workers=num_workers)
        ref_data = manif_metric.extract_features(test_loader, transform=transformer(scale=1), device=device, dtype=dtype)
        manif_metric.compute_ref_stats(ref_data, k=knn)

        ### Train loader
        train_loader = DataLoader(dataset=dataset, sampler=rng.permutation(dataset.train_set)[:gen_size], batch_size=batch_size, num_workers=num_workers)
        for si, scale in enumerate(scales):
            pbar.update(1)
            pbar.set_description(f'Metric Run={run} Scale={scale:.2f}')

            ### Draw transformed samples
            if run == 0:
                if si == 0:
                    draw_tensor = list()
                    for data in train_loader:
                        draw_tensor.append(data[0] if isinstance(data, (tuple, list)) else data)
                        if len(draw_tensor) * batch_size >= 10:
                            break
                    draw_tensor = torch.concat(draw_tensor, dim=0)[:10]
                collect.append(transformer(scale)(draw_tensor).permute(0, 2, 3, 1).cpu().numpy())
                if si == len(scales) - 1:
                    block_draw(np.stack(collect, axis=1), os.path.join(save_dir, f'{type(dataset).__name__}_transform_samples.png'), border=True)

            ### Generated data
            gen_data = manif_metric.extract_features(train_loader, transform=transformer(scale=scale), device=device, dtype=dtype)
            manif_metric.compute_gen_stats(gen_data, k=knn)
            
            ### Compute metrics
            precs = manif_metric.sym_precision()
            recalls = manif_metric.sym_recall()
            
            metrics['precision'][si, run] = precs.precision
            metrics['comp_precision'][si, run] = precs.comp_precision
            metrics['sym_precision'][si, run] = precs.sym_precision
            
            metrics['recall'][si, run] = recalls.recall
            metrics['coverage'][si, run] = recalls.coverage
            metrics['sym_recall'][si, run] = recalls.sym_recall

            ### Conserve memory
            del gen_data
            manif_metric.gen_stats = None
    pbar.close()

    ### Save and plot
    title=f'{type(dataset).__name__}'
    xlabel='Contrast Scale'
    with open(os.path.join(save_dir, f'{title.lower().replace(" ", "_")}_metrics.json'), 'w+') as fs:
        json.dump(dict(scales=scales, metrics=dict(metrics)), fs, cls=NumpyEncoder, indent=2)

    colors = matplotlib.colormaps['plasma'](np.linspace(0., .8, 3))
    markers = ['.', 's', 'v', '^', 'd', 'x', '1', '2', '3', '4']
    linestyles = ['--', '--', None]
    
    metric_groups = {
        'precision': ['precision', 'comp_precision', 'sym_precision'],
        'recall': ['recall', 'coverage', 'sym_recall']}
    
    mean_metrics = dict()
    std_metrics = dict()
    for group_name, metric_group in metric_groups.items():
        fig = plt.figure(figsize=(8,6), constrained_layout=True)
        ax = fig.add_subplot()
        xticks = np.arange(len(scales))
        for ni, name in enumerate(metric_group):
            metric = metrics[name]
            mean_metric = np.mean(metric, axis=1)
            std_metric = np.std(metric, axis=1)
            ax.plot(xticks, mean_metric,
                label=METRIC_STR[name], color=colors[ni], marker=markers[ni], alpha=0.8, linestyle=linestyles[ni])
            ax.fill_between(xticks, mean_metric-std_metric, mean_metric+std_metric,
                color=colors[ni], alpha=0.2)
            mean_metrics[name] = mean_metric
            std_metrics[name] = std_metric
        ax.legend(loc=8)
        ax.grid(True, which='both', linestyle='dotted')
        
        ax.set_xticks(xticks)
        xticks_labels = [' '] * len(xticks)
        xticks_labels[0] = f'{scales[0]:0.1f}'
        xticks_labels[len(xticks)//2] = f'{scales[len(xticks)//2]:0.1f}'
        xticks_labels[-1] = f'{scales[-1]:0.1f}'
        ax.set_xticklabels(xticks_labels)

        ax.set_ylim(bottom=0, top=1.2)
        ### Hide top yticklabel
        yticks = np.arange(0, 1.4, 0.2)
        yticklabels = [f'{ytick:0.1f}' for ytick in yticks]
        yticklabels[-1] = ''
        ax.set_yticks(yticks, yticklabels)

        ### Setup titles
        ax.set_title(f'{title} {METRIC_STR[group_name]}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(METRIC_STR[group_name])

        fig.savefig(os.path.join(save_dir, f'{title.lower().replace(" ", "_")}_{group_name}.pdf'))

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True, help=f'Choose from {list(EXP_DICT.keys())}')
    parser.add_argument('--save_dir', required=True, help='Directory to save results.')
    parser.add_argument('--dataset', default='cifar10', help=f'Choose from celeba or cifar10.')
    parser.add_argument('--data_size', type=int, help=f'Number of reference and generated samples.')
    parser.add_argument('--ref_size', type=int, help=f'Number of reference samples (overrides data_size).')
    parser.add_argument('--gen_size', type=int, help=f'Number of generated samples (overrides data_size).')
    parser.add_argument('--ref_r', type=float, help=f'Reference radius or half-edge length.')
    parser.add_argument('--lower', type=float, help=f'Lower generated radius or half-edge length or scale.')
    parser.add_argument('--upper', type=float, help=f'Upper generated radius or half-edge length or scale.')
    parser.add_argument('--step', type=float, help=f'Step of generated radius or half-edge length or scale.')
    parser.add_argument('--knn', type=int, help=f'K of KNN for metrics computation.')
    parser.add_argument('--dims', type=int, nargs='+', help=f'List of data dimensions to consider in toy experiments.')
    parser.add_argument('--batch_size', type=int, help=f'Batch size for extracting features.')
    parser.add_argument('--num_workers', type=int, help=f'Number of workers for reading data.')
    parser.add_argument('--num_features', type=int, help=f'Number of features to select from the extractor model.')
    parser.add_argument('--pretrained', default=1, type=bool, help=f'Whether to load pretrained weights into the feature extractor, choose 0 or 1.')
    parser.add_argument('--seed', default=1000, type=int, help=f'Random seed controlling the data generation or selection (incremented by each run).')
    parser.add_argument('--runs', default=1, type=int, help=f'Number of random runs.')
    return parser.parse_args()

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

EXP_DICT = {
    'hypersphere_exp': hypersphere_exp,
    'hypercube_exp': hypercube_exp,
    'normal_hypersphere_exp': normal_hypersphere_exp,
    'feature_scaled_exp': feature_scaled_exp,
    'contrast_exp': contrast_exp,
}

if __name__ == '__main__':
    ### Plot setup
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    SMALL_SIZE = 20
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    ### Run setup
    args = setup_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(os.path.join(args.save_dir, 'terminal.txt'))
    setup_seed(args.seed)
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'\n>>> Starting run on device {args.device}\n')
    
    if args.exp in ['feature_scaled_exp', 'contrast_exp']:    
        ### Extractor model setup
        args.model = VGG16(pretrained=args.pretrained, num_features=args.num_features)
        
        ### Dataset setup
        if args.dataset == 'celeba':
            dataset = CelebADataset(128)
            dataset.draw(DataLoader(dataset, sampler=dataset.test_set[:25]),
                os.path.join(args.save_dir, 'celeba128_samples.png'))
        elif args.dataset == 'cifar10':
            dataset = CIFAR10Dataset()
            dataset.draw(DataLoader(dataset, sampler=dataset.test_set[:25]),
                os.path.join(args.save_dir, 'cifar10_samples.png'))
        else:
            raise ValueError(f'Dataset name {args.dataset} undefined!')
        args.dataset = dataset

    ### Run experiment
    EXP_DICT[args.exp](**{k: val for k, val in vars(args).items() if val is not None})
    logger.close()
