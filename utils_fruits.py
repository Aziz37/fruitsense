from torchvision import transforms
import math
import torch


class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    


def get_transforms(split, aug, num_channel=8):
    mean = (0.5,) * num_channel
    std = (0.5,) * num_channel
    normalize = transforms.Normalize(mean, std)
    if split == 'train':
        aug_list = aug.split(',')
        transforms_list = []

        if 'crop' in aug_list:
          transforms_list.append(transforms.RandomResizedCrop(size=50, scale=(0.2, 1.)))

        if 'flip' in aug_list:
          transforms_list.append(transforms.RandomHorizontalFlip())

        if 'rotate' in aug_list:
          transforms_list.append(transforms.RandomRotation(degrees=(-15, 15)))

        transforms_list.append(normalize)
        transform = transforms.Compose(transforms_list)
    else:
        transform = transforms.Compose([
          # transforms.Resize(256),
          # transforms.CenterCrop(224),
          normalize,
        ])

    return transform

def get_label_dim(dataset):
    if dataset in ['FruitsDataset', 'FruitsDatasetV2', 'FruitsDatasetRGB', 'FruitsDataset30C']:
        label_dim = 1
    else:
        raise ValueError(dataset)
    return label_dim


class AverageMeter(object):
    def __init__(self):
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


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    eta_min = lr * (args.lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def set_optimizer(opt, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate,
                                momentum=opt.momentum, weight_decay=opt.weight_decay)

    return optimizer
