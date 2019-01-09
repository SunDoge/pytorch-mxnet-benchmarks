from models.resent20_mxnet import resnet20_cifar
import argparse
from mxnet import gluon, autograd
from mxnet.gluon.data.vision import transforms as T, datasets
import mxnet as mx
import os
import time
import shutil


parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--epochs', default=200, type=int,
                    metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.1,
                    type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9,
                    type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4,
                    type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate',
                    action='store_true', help='evaluate model on validation set')
parser.add_argument('--hybridize', action='store_true')
args = parser.parse_args()


class RandomCrop(gluon.nn.Block):
    def __init__(self, size, padding=0):
        super().__init__()
        self.size = size
        self.padding = padding

    def forward(self, x):
        x = mx.image.copyMakeBorder(
            x, self.padding, self.padding, self.padding, self.padding)
        # print(x.shape)
        x, _ = mx.image.random_crop(x, (self.size, self.size))
        return x


normalize = T.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

train_transfrom = T.Compose([
    RandomCrop(32, padding=4),
    # T.RandomResizedCrop(32),
    T.RandomFlipLeftRight(),
    T.ToTensor(),
    normalize
])

val_transform = T.Compose([
    T.ToTensor(),
    normalize
])

trainset = datasets.CIFAR10(
    './data', train=True).transform_first(train_transfrom)
trainloader = gluon.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(
    './data', train=False).transform_first(val_transform)
testloader = gluon.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

best_prec = 0


class AverageMeter:

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


def adjust_learning_rate(trainer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""

    if epoch < 80:
        lr = args.lr
    elif epoch < 120:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01

    trainer.set_learning_rate(lr)


def train(trainloader, net, loss_fn, trainer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    train_metric.reset()

    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        data_time.update(time.time() - end)
        # print(batch)
        input, target = input.as_in_context(
            ctx), target.as_in_context(ctx)

        with autograd.record():
            output = net(input)
            loss = loss_fn(output, target)

        losses.update(loss.mean().asscalar(), input.shape[0])
        train_metric.update(target, output)
        _, acc = train_metric.get()
        acc *= 100
        top1.update(acc.item(), input.shape[0])

        loss.backward()
        trainer.step(input.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                      epoch, i, len(trainloader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, net, loss_fn):
    batch_time = AverageMeter()
    losses = AverageMeter()
    metric = mx.metric.Accuracy()
    top1 = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.as_in_context(
            ctx), target.as_in_context(ctx)

        output = net(input)
        loss = loss_fn(output, target)

        losses.update(loss.mean().asscalar(), input.shape[0])
        metric.update(target, output)
        _, acc = metric.get()
        acc *= 100
        top1.update(acc.item(), input.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return top1.val


def save_checkpoint(net, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.params')
    net.save_parameters(filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.params'))


if __name__ == '__main__':
    if not os.path.exists('result'):
        os.makedirs('result')
    fdir = 'result/resnet20_cifar10_mxnet'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    ctx = mx.gpu()
    net = resnet20_cifar()
    net.initialize(init=mx.initializer.Xavier(), ctx=ctx)
    optimizer = mx.optimizer.SGD(
        momentum=args.momentum, learning_rate=args.lr, wd=args.weight_decay)
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    train_metric = mx.metric.Accuracy()

    trainer = gluon.Trainer(net.collect_params(), optimizer)

    if args.hybridize:
        net.hybridize()

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint "{}"'.format(args.resume))
            net.load_parameters(args.resume, ctx=ctx)
            print("=> loaded checkpoint '{}'".format(
                args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        pass
        exit(0)

    epoch_start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(trainer, epoch)

        train(trainloader, net, loss_fn, trainer, epoch)

        prec = validate(testloader, net, loss_fn)

        is_best = prec > best_prec
        best_prec = max(prec, best_prec)

        save_checkpoint(net, is_best, fdir)

    epoch_end = time.time()
    print('total time:', epoch_end - epoch_start)
    print('best acc:', best_prec)
