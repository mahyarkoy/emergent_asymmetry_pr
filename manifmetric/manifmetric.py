from tqdm import tqdm
import sys
import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.transforms import functional as FT, InterpolationMode

### Utility
class DotDict(dict): 
    '''
    Dictionary with access using attributes.
    '''
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        del self[key]

class Logger(object):
    def __init__(self, log_path, log_option='a+', force_flush=True):
        '''
        Logging from stdout and stderr into stdout and log file simultaneously.
        '''
        self.log = open(log_path, log_option)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.force_flush = force_flush
        
        ### Take control from stdout and stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, message):
        if len(message) == 0:
            return
        self.log.write(message)
        self.stdout.write(message)
        if self.force_flush:
            self.flush()

    def flush(self):
        self.log.flush()
        self.stdout.flush()

    def close(self):
        self.flush()
        
        ### Return control to stdout and stdin
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr
        
        self.log.close()
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    def __del__(self):
        self.close()

### Extractor model VGG16
class VGG16(nn.Module):
    def __init__(self, pretrained=True, num_features=None):
        super().__init__()
        try:
            from torchvision.models import VGG16_Weights
            self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1) if pretrained else vgg16()
        except ImportError:
            self.vgg = vgg16(pretrained=pretrained)
        self.num_features = num_features
    def forward(self, in_):
        in_t = FT.resize(in_, 256, interpolation=InterpolationMode.BILINEAR, antialias=None)
        in_t = FT.center_crop(in_t, 224)
        in_t = FT.normalize(in_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        pre_fc = self.vgg.features(in_t).flatten(1)
        features = self.vgg.classifier[:4](pre_fc)
        return features[:, :self.num_features] if self.num_features is not None else features

### Metrics
class ManifoldMetric:
    def __init__(self, model=None, ref_data=None, gen_data=None, k=None):
        '''
        Class to compute precison, comp_precision, recall, coverage, density, sym_precision, sym_recall.
        @model: model to use for embedding, can be string 'vgg16' or a function.
        @ref_data: encoded vectors of shape (b, ...) of type tensor or ndarray.
        @gen_data: encoded vectors of shape (b, ...) of type tensor or ndarray.
        @k: k in k-nearest neighbor neighborhood computation.
        '''
        self.ref_stats = self.compute_ref_stats(ref_data, k=k) if ref_data is not None else None
        self.gen_stats = self.compute_gen_stats(gen_data, k=k) if gen_data is not None else None
        if model is not None and isinstance(model, str):
            self.model = {'vgg16': VGG16}[model]()
        else:
            self.model = model
    
    @torch.no_grad()
    def extract_features(self, dataloader, device, transform=None, num_samples=None,
        output_shape=None, output_ids=None, dtype=torch.float32):
        '''
        Extracts features using self.model applied to the given dataloader.
        @dataloader: data should be of shape (b, c, h, w) in [0, 1] dynamic range, use transform to adjust.
        @transform: a function applied right after reading the data from dataloader and right before feeding it to the model.
        @num_samples: number of samples to keep from the dataloader.
        @output_shape: the lenght of the zero template to fill with features, useful for multiprocessing.
        @output_ids: list of where to put the extracted features on the output zero template of shape (output_shape, dim).
        @dtype: dtype of the input and output of the model.
        '''
        model = self.model
        transform = transform if transform is not None else lambda x: x
        
        model.eval()
        model.to(device)
        features = list()
        feature_count = 0
        for data in dataloader:
            in_tensor = torch.as_tensor(transform(data), dtype=dtype, device=device)
            out = model(in_tensor)
            features.append(out[0] if isinstance(out, (list, tuple)) else out)
            feature_count += features[-1].shape[0]
            if num_samples is not None and feature_count >= num_samples:
                break
        
        features = torch.cat(features)[:num_samples] if num_samples is not None else torch.cat(features)
        if output_shape is not None and output_ids is not None:
            assert len(output_ids) == features.shape[0], f'>>> {len(output_ids)} != {features.shape}'
            output = torch.zeros([output_shape, *features.shape[1:]], dtype=features.dtype, device=device)
            output[output_ids] = features
        else:
            output = features
        
        return output
    
    @torch.no_grad()
    def compute_stats(self, data, k):
        '''
        Computes stats required for metric computation.
        @data: tensor or ndarray of shape (b, ...).
        @k: number of neighbors to calculate and keep.
        @return: dict with keys {data, dist2, dist2_args}.
        '''
        data = torch.as_tensor(data).flatten(1)
        dist2, dist2_args = self.compute_knn(data, data, k+1)
        return DotDict(data=data, dist2=dist2, dist2_args=dist2_args)
        
    def compute_ref_stats(self, data, k):
        '''
        Computes and assigns reference data stats required for metric computation.
        @data: tensor or ndarray of shape (b, ...).
        @k: number of neighbors to calculate and keep, if None, will not compute knn stats dist2 and dist2_args.
        @return: dict with keys {data, dist2, dist2_args, mu, cov}.
        '''
        self.ref_stats = self.compute_stats(data, k)
        return self.ref_stats
    
    def compute_gen_stats(self, data, k):
        '''
        Computes and assigns generated data stats required for metric computation.
        @data: tensor or ndarray of shape (b, ...).
        @k: number of neighbors to calculate and keep, if None, will not compute knn stats dist2 and dist2_args.
        @return: dict with keys {data, dist2, dist2_args, mu, cov}.
        '''
        self.gen_stats = self.compute_stats(data, k)
        return self.gen_stats
    
    @torch.no_grad()
    def compute_knn(self, source, target, k):
        '''
        Computes the distance and index of the k nearest neighbor of each element
        of source in target.
        @source: tensor or ndarray of shape (b, ...).
        @target: tensor or ndarray of shape (b, ...).
        @k: number of neighbors to calculate and keep.
        @return: tensor or ndarray of shape (b, k) containing distance^2 and indicies.
        '''
        source = torch.as_tensor(source)
        target = torch.as_tensor(target, dtype=source.dtype, device=source.device)
        size = source.shape[0]
        k = min(k, target.shape[0])
        assert k > 0

        dist2 = torch.as_tensor(torch.zeros((size, k)), dtype=source.dtype, device=source.device)
        dist2_args = torch.as_tensor(torch.zeros((size, k)), dtype=torch.int64, device=source.device)
        for i in range(size):
            sort_dist2, sort_args = torch.sort((source[i] - target).pow(2).flatten(1).sum(-1))
            dist2[i] = sort_dist2[:k]
            dist2_args[i] = sort_args[:k]
        return dist2, dist2_args
    
    @torch.no_grad()
    def coverage(self, ref_stats=None, gen_stats=None, k=5):
        '''
        Returns fraction of ref_data that have at least one point of gen_data within their KNN.
        @ref_stats: dict output of self.compute_stats, if None, will use self.ref_stats.
        @gen_stats: dict output of self.compute_stats, if None, will use self.gen_stats.
        @k: k in k-nearest neighbor neighborhood computation, must be <= than k used in self.compute_stats.
        '''
        ref_stats = self.ref_stats if ref_stats is None else ref_stats
        gen_stats = self.gen_stats if gen_stats is None else gen_stats

        gen_data = torch.as_tensor(gen_stats.data, dtype=ref_stats.data.dtype, device=ref_stats.data.device)
        collect = 0
        for di, d in enumerate(ref_stats.data):
            dist2 = (d - gen_data).pow(2).flatten(1).sum(-1)
            collect += torch.any(dist2 < ref_stats.dist2[di, k])
        return collect.item() / ref_stats.data.shape[0]

    @torch.no_grad()
    def density(self, ref_stats=None, gen_stats=None, k=5):
        '''
        Returns average number of KNN neighborhoods of ref_data that contain each gen_data point.
        @ref_stats: dict output of self.compute_stats, if None, will use self.ref_stats.
        @gen_stats: dict output of self.compute_stats, if None, will use self.gen_stats.
        @k: k in k-nearest neighbor neighborhood computation, must be <= than k used in self.compute_stats.
        '''
        ref_stats = self.ref_stats if ref_stats is None else ref_stats
        gen_stats = self.gen_stats if gen_stats is None else gen_stats

        gen_data = torch.as_tensor(gen_stats.data, dtype=ref_stats.data.dtype, device=ref_stats.data.device)
        collect = 0
        for d in gen_data:
            dist2 = (d - ref_stats.data).pow(2).flatten(1).sum(-1)
            collect += torch.sum(dist2 < ref_stats.dist2[:, k])
        return collect.item() / (gen_data.shape[0] * k)
    
    @torch.no_grad()
    def precision(self, ref_stats=None, gen_stats=None, k=5):
        '''
        Returns fraction of gen_data that are within the KNN of at least one ref_data.
        @ref_stats: dict output of self.compute_stats, if None, will use self.ref_stats.
        @gen_stats: dict output of self.compute_stats, if None, will use self.gen_stats.
        @k: k in k-nearest neighbor neighborhood computation, must be <= than k used in self.compute_stats.
        '''
        ref_stats = self.ref_stats if ref_stats is None else ref_stats
        gen_stats = self.gen_stats if gen_stats is None else gen_stats

        gen_data = torch.as_tensor(gen_stats.data, dtype=ref_stats.data.dtype, device=ref_stats.data.device)
        collect = 0
        for d in gen_data:
            dist2 = (d - ref_stats.data).pow(2).flatten(1).sum(-1)
            collect += torch.any(dist2 < ref_stats.dist2[:, k])
        return collect.item() / gen_data.shape[0]
    
    @torch.no_grad()
    def recall(self, ref_stats=None, gen_stats=None, k=5):
        '''
        Returns fraction of ref_data that are within the KNN of at least one gen_data.
        @ref_stats: dict output of self.compute_stats, if None, will use self.ref_stats.
        @gen_stats: dict output of self.compute_stats, if None, will use self.gen_stats.
        @k: k in k-nearest neighbor neighborhood computation, must be <= than k used in self.compute_stats.
        '''
        ref_stats = self.ref_stats if ref_stats is None else ref_stats
        gen_stats = self.gen_stats if gen_stats is None else gen_stats
        return self.precision(ref_stats=gen_stats, gen_stats=ref_stats, k=k)
    
    @torch.no_grad()
    def comp_precision(self, ref_stats=None, gen_stats=None, k=5):
        '''
        Returns fraction of data that have at least one point of self.data within their KNN.
        @ref_stats: dict output of self.compute_stats, if None, will use self.ref_stats.
        @gen_stats: dict output of self.compute_stats, if None, will use self.gen_stats.
        @k: k in k-nearest neighbor neighborhood computation, must be <= than k used in self.compute_stats.
        '''
        ref_stats = self.ref_stats if ref_stats is None else ref_stats
        gen_stats = self.gen_stats if gen_stats is None else gen_stats
        return self.coverage(ref_stats=gen_stats, gen_stats=ref_stats, k=k)
    
    def sym_precision(self, ref_stats=None, gen_stats=None, k=5):
        '''
        Returns dict of (sym_precision, precision, comp_precision).
        @ref_stats: dict output of self.compute_stats, if None, will use self.ref_stats.
        @gen_stats: dict output of self.compute_stats, if None, will use self.gen_stats.
        @k: k in k-nearest neighbor neighborhood computation, must be <= than k used in self.compute_stats.
        '''
        prec = self.precision(ref_stats, gen_stats, k)
        comp_prec = self.comp_precision(ref_stats, gen_stats, k)
        return DotDict(sym_precision=min(prec, comp_prec), precision=prec, comp_precision=comp_prec)
    
    def sym_recall(self, ref_stats=None, gen_stats=None, k=5):
        '''
        Returns dict of (sym_recall, recall, and coverage).
        @ref_stats: dict output of self.compute_stats, if None, will use self.ref_stats.
        @gen_stats: dict output of self.compute_stats, if None, will use self.gen_stats.
        @k: k in k-nearest neighbor neighborhood computation, must be <= than k used in self.compute_stats.
        '''
        recall = self.recall(ref_stats, gen_stats, k)
        coverage = self.coverage(ref_stats, gen_stats, k)
        return DotDict(sym_recall=min(recall, coverage), recall=recall, coverage=coverage)