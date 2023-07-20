import numpy as np
import matplotlib
import pickle as pk
import os
import glob
from PIL import Image
import torch

class CIFAR10Dataset:
    def __init__(self, **kwargs):
        '''
        Reads CIFAR10 images of shape (b, 32, 32, 3) in [0, 1].
        '''
        super().__init__(**kwargs)
        self.root_dir = 'data/cifar'
        data_paths = glob.glob(os.path.join(self.root_dir, 'cifar-10-batches-py/data_batch_*'))
        data_paths.append(os.path.join(self.root_dir, 'cifar-10-batches-py/test_batch'))
        img_collect = list()
        lab_collect = list()
        for data_path in data_paths:
            with open(data_path, 'rb') as fs:
                datadict = pk.load(fs, encoding='latin-1')
            images = np.transpose(datadict['data'].reshape((-1, 3, 32, 32)), axes=(0, 2, 3, 1))
            labels = np.array(datadict['labels'])
            img_collect.append(images / 255.)
            lab_collect.append(labels)
        self.data = np.concatenate(img_collect, axis=0)
        self.labels = np.concatenate(lab_collect, axis=0)
        assert np.all(self.data.shape == (60000, 32, 32, 3))
        assert np.all(self.labels.shape == (60000,))

        self.train_set = np.arange(50000)
        self.test_set = np.arange(50000, 60000)
        with open(os.path.join(self.root_dir, 'cifar-10-batches-py/batches.meta'), 'rb') as fs:
            metadict = pk.load(fs, encoding='latin-1')
            self.id_to_class = {key: val for key, val in enumerate(metadict['label_names'])}
            self.class_to_id = {val: key for key, val in self.id_to_class.items()}
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def draw(self, data_loader, save_path):
        collect = list()
        for data in data_loader:
            collect.append(data[0])
        collect = torch.cat(collect).cpu().numpy()
        block_draw(collect, save_path, border=True)

class CelebADataset:
    def __init__(self, img_size=128, **kwargs):
        '''
        Reads CelebA images of shape (b, img_size, img_size, 3) in [0, 1].
        '''
        super().__init__(**kwargs)
        self.img_size = img_size
        self.root_dir = 'data/celeba'
        self.files = sorted(glob.glob(os.path.join(self.root_dir, 'img_align_celeba/*.jpg')))
        self.train_set = list()
        self.test_set = list()
        with open(os.path.join(self.root_dir, 'list_eval_partition.txt'), 'r') as fs:
            split_dict = dict()
            for l in fs:
                split_dict[l.strip().split(' ')[0]] = l.strip().split(' ')[-1]
        for fi, file in enumerate(self.files):
            label = split_dict[file.split('/')[-1]]
            if label in ['0', '1']:
                self.train_set.append(fi)
            else:
                self.test_set.append(fi)
    
    def __getitem__(self, idx):
        return read_image(self.files[idx], self.img_size, sqcrop=True, center_crop=(121, 89), crop_size=128)
    
    def draw(self, data_loader, save_path):
        collect = list()
        for data in data_loader:
            collect.append(data)
        collect = torch.cat(collect).cpu().numpy()
        block_draw(collect, save_path, border=True)

def read_image(img_path, img_size=None, sqcrop=False, bbox=None, verbose=False, center_crop=None, crop_size=None):
    '''
    Reads single image from path.
    @img_path: path to image file.
    @img_size: the width and height used to resize the image (width and height of the returned image).
    @sqcrop: whether to crop the image using center_crop and crop_size.
    @bbox: bounding box in (left, top, right, bottom) format to use for cropping, if sqcrop is False.
    @verbose: if True, returns tuple of (image, original_width, original_height).
    @center_crop: the center around which to crop a square of crop_size, if None, uses the center of original image.
    @crop_size: the width and height to crop around center_crop, if None, uses minimum of original width and height.
    @return: ndarray of shape (img_size, img_size, 3) and in [0., 1.] dynamic range.
    '''
    with open(img_path, 'rb') as fs:
        img = Image.open(fs)
        img.load()

    w, h = img.size
    w_orig = w
    h_orig = h

    ### Crop images
    if sqcrop:
        crop_size = min(w, h) if crop_size is None else crop_size
        cy, cx = (h//2, w//2) if center_crop is None else center_crop
        left = min(cx - crop_size//2, 0)
        top = min(cy - crop_size//2, 0)
        right = max(left + crop_size, w)
        bottom = max(top + crop_size, h)
        img_sq = img.crop((left, top, right, bottom))
    
    elif bbox is not None:
        left = bbox[0]
        top = bbox[1]
        right = bbox[2]
        bottom = bbox[3]
        img_sq = img.crop((left, top, right, bottom))
    
    else:
        img_sq = img
    
    ### Resize images
    w, h = img_sq.size
    if img_size is not None and (w != img_size or h != img_size):
        img_re = img_sq.resize((img_size, img_size), Image.BILINEAR)
        w, h = img_size, img_size
    else:
        img_re = img_sq

    ### Convert to numpy and to [0, 1] dynamic range
    img_re = np.array(img_re.getdata())
    ### Next line is because pil removes the channels for black and white images!
    img_re = img_re if len(img_re.shape) > 1 else np.repeat(img_re[..., np.newaxis], 3, axis=1)
    img_re = img_re.reshape((h, w, -1))
    img_o = (img_re / 255.0)
    img_o = img_o[:, :, :3]
    return img_o if not verbose else (img_o, w_orig, h_orig)

def block_draw(imgs, path, border=False, rows=None, cols=None):
    '''
    @imgs: ndarray of shape (rows, cols, h, w, c) | (b, h, w, c) | (h, w ,c) | (h, w) with values in [0,1]. If c is not 3 then draws first channel only.
    @path: save file path.
    @border: whether to add white border to each image.
    @rows: if not none, will plot rows*cols images.
    @cols: if not none, will plot rows*cols images.
    Note: if both rows and cols are None, then cols is sqrt(b) and rows is chosen to include all images.
    Note: missing images to form rows*cols arrangement are filled with white.
    '''
    if imgs.ndim == 5:
        rows, cols, imh, imw, imc = imgs.shape

    elif imgs.ndim == 4:
        imb, imh, imw, imc = imgs.shape
        if rows is None and cols is None:
            cols = int(np.sqrt(imb))
            rows = int(np.ceil(imb / cols))
        elif rows is None:
            rows = int(np.ceil(imb / cols))
        elif cols is None:
            cols = int(np.ceil(imb / rows))

        white_img = np.ones([imh, imw, imc])
        imgs = np.concatenate([img for img in imgs]+[white_img]*(rows*cols-imb), axis=0)
        imgs = imgs.reshape([rows, cols, imh, imw, imc])
    
    elif imgs.ndim in [2, 3]:
        if imgs.ndim == 2:
            imh, imw = imgs.shape
            imc = 1
        else:
            imh, imw, imc = imgs.shape
        rows = cols = 1
        imgs = imgs.reshape([rows, cols, imh, imw, imc])
    
    else:
        raise ValueError(f'Cannot draw images with shape {imgs.shape}')
    
    imgs = np.transpose(imgs, [1, 0, 2, 3, 4])
    imgs = imgs if imc == 3 else np.repeat(imgs[:,:,:,:,:1], 3, axis=4)
    imc = 3
    
    if border:
        imgs = add_color_borders(imgs.reshape([-1, imh, imw, imc]), 
            .5*np.ones(cols*rows), max_label=0., color_map='RdBu')
        imb, imh, imw, imc = imgs.shape
    
    if cols > 1:
        imgs = imgs.reshape([cols, rows*imh, imw, imc])
        imgs = np.concatenate([img for img in imgs], axis=1)
    else:
        imgs = imgs.reshape([-1, imw, imc])
    imgs = np.clip(np.rint(imgs * 255.0), 0.0, 255.0).astype(np.uint8)
    Image.fromarray(imgs, 'RGB').save(path)
    return imgs

def add_color_borders(imgs, labels, max_label=None, color_map=None, color_set=None, fh=2, fw=2):
    '''
    Adds a color border to imgs corresponding to its im_label.
    @imgs: ndarray of shape (b, h, w, c) with values in [0,1].
    @labels: ndarray of shape (b) with int values.
    @max_label: if not None, max_label+1 will be used to normalize labels for access into color_map.
    @color_map: if not None, color_map to select from based on labels.
    @color_set: if color_map is None, ndarray of shape (max_label+1, 3) to select from based on labels.
    @fh: the border size along image height.
    @fw: the border size along image width.
    '''
    imb, imh, imw, imc = imgs.shape
    max_label = labels.max() if max_label is None else max_label
    imgs = np.repeat(imgs, 3, axis=-1) if imc == 1 else imgs
    
    ### Pick RGB color for each label: (imb, 3) in [-1,1]
    if color_map is None:
        assert color_set is not None
        rgb_template = color_set[labels, ...][:, :3]
    else:
        cmap = matplotlib.colormaps[color_map]
        labels_norm = labels.reshape([-1]).astype(np.float32) / (max_label + 1)
        rgb_template = cmap(labels_norm)[:, :3]
    rgb_template = np.tile(rgb_template.reshape((imb, 1, 1, 3)), (1, imh+2*fh, imw+2*fw, 1))

    ### Put imgs into rgb_template
    for b in range(imb):
        rgb_template[b, fh:imh+fh, fw:imw+fw, :] = imgs[b, ...]

    return rgb_template