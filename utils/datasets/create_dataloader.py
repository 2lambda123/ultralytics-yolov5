from codecs import ignore_errors
import torch
import os
from torch.utils.data import DataLoader, distributed
# #####################################
from utils.general import LOGGER
from utils.torch_utils import torch_distributed_zero_first
from .LoadImagesAndLabels import LoadImagesAndLabels
from .InfiniteDataLoader import InfiniteDataLoader
from .collate_fns import collate_fn4, collate_fn
# #####################################


def create_dataloader(path, imgsz, batch_size, stride, 
    single_cls=False, hyp=None, augment=False, cache=False, 
    pad=0.0, rect=False, rank=-1, workers=8, image_weights=False, 
    quad=False, prefix='', shuffle=False, ignore_cache: bool=False):
    """
        rect: use rectangular training
        quad: use quad dataloader

    """
    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False

    # init dataset *.cache only once if DDP
    with torch_distributed_zero_first(rank):  
        dataset = LoadImagesAndLabels(
            path=path, 
            img_size=imgsz, 
            batch_size=batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            ignore_cache=ignore_cache,
            )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=collate_fn4 if quad else collate_fn), dataset
