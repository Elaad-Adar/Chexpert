# ref: https://github.com/pytorch/examples/blob/792d336019a28a679e29cf174e10cee80ead8722/imagenet/main.py#L363

import torch
from math import log, ceil, floor

def closest_power(x):
    possible_results = floor(log(x, 2)), ceil(log(x, 2))
    p = min(possible_results, key=lambda z: abs(x - 2 ** z))

    return 2 ** p

def writer_add_scalars(writer, prefix, kvs, epoch):
    for k, v in kvs.items():
        writer.add_scalar(f"{prefix}/{k}", v, epoch)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\n' + '\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(max(topk), 5)
        batch_size = target.size(0)
        # output.topk(maxk = the k in “top-k”, 1 – the dimension to sort along ,largest=True, sorted=True)
        # outputs values tensor (_) and indeces tensor (pred)
        _, pred = output.topk(maxk, 1, True, True)
        # pred = pred.t()
        correct = pred.eq(target.expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
