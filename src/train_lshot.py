import logging
import os
import random
import shutil
import time
import collections
import pickle
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR
import tqdm
from scipy.stats import mode
from utils import configuration
from lshot_update import bound_update
from numpy import linalg as LA
import datasets
import models
from scipy import sparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
best_prec1 = -1


def main():
    global args, best_prec1
    args = configuration.parser_args()
    ### initial logger
    log = setup_logger(args.save_path + args.log_file)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    cudnn.deterministic = True
    # create model
    log.info("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=args.num_classes, remove_linear=args.do_meta_train)

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    if args.label_smooth > 0:
        criterion = SmoothCrossEntropy(epsilon=args.label_smooth).cuda()

    else:
        criterion = nn.CrossEntropyLoss().cuda()

    optimizer = get_optimizer(model)

    if args.pretrain:
        pretrain = args.pretrain + '/checkpoint.pth.tar'
        if os.path.isfile(pretrain):
            log.info("=> loading pretrained weight '{}'".format(pretrain))
            checkpoint = torch.load(pretrain)
            model_dict = model.state_dict()
            params = checkpoint['state_dict']
            params = {k: v for k, v in params.items() if k in model_dict}
            model_dict.update(params)
            model.load_state_dict(model_dict)
        else:
            log.info('[Attention]: Do not find pretrained model {}'.format(pretrain))

    # resume from an exist checkpoint
    if os.path.isfile(args.save_path + '/checkpoint.pth.tar') and args.resume == '':
        args.resume = args.save_path + '/checkpoint.pth.tar'

    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # scheduler.load_state_dict(checkpoint['scheduler'])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info('[Attention]: Do not find checkpoint {}'.format(args.resume))

    # Data loading code

    if args.evaluate:
        do_extract_and_evaluate(model, log)
        return

    args.enlarge = False
    if args.do_meta_train:
        sample_info = [args.meta_train_iter, args.meta_train_way, args.meta_train_shot, args.meta_train_query]
        train_loader = get_dataloader('train', not args.disable_train_augment, sample=sample_info)
    else:
        train_loader = get_dataloader('train', not args.disable_train_augment, shuffle=True)

    sample_info = [args.meta_val_iter, args.meta_val_way, args.meta_val_shot, args.meta_val_query]
    val_loader = get_dataloader('val', False, sample=sample_info)

    scheduler = get_scheduler(len(train_loader), optimizer)
    tqdm_loop = warp_tqdm(list(range(args.start_epoch, args.epochs)))
    for epoch in tqdm_loop:
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, scheduler, log)
        scheduler.step(epoch)
        # evaluate on meta validation set
        is_best = False
        if (epoch + 1) % args.meta_val_interval == 0:
            prec1 = meta_val(val_loader, model)
            log.info('Meta Val {}: {}'.format(epoch, prec1))
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if not args.disable_tqdm:
                tqdm_loop.set_description('Best Acc {:.2f}'.format(best_prec1 * 100.))

        # remember best prec@1 and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            # 'scheduler': scheduler.state_dict(),
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, folder=args.save_path)

    # do evaluate at the end
    args.enlarge = True
    do_extract_and_evaluate(model, log)


def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


def metric_prediction(gallery, query, train_label, metric_type):
    gallery = gallery.view(gallery.shape[0], -1)
    query = query.view(query.shape[0], -1)
    distance = get_metric(metric_type)(gallery, query)
    predict = torch.argmin(distance, dim=1)
    predict = torch.take(train_label, predict)

    return predict


def meta_val(test_loader, model, train_mean=None):
    top1 = AverageMeter()
    model.eval()

    with torch.no_grad():
        tqdm_test_loader = warp_tqdm(test_loader)
        for i, (inputs, target) in enumerate(tqdm_test_loader):
            target = target.cuda(0, non_blocking=True)
            output = model(inputs, True)[0].cuda(0)
            if train_mean is not None:
                output = output - train_mean
            train_out = output[:args.meta_val_way * args.meta_val_shot]
            train_label = target[:args.meta_val_way * args.meta_val_shot]
            test_out = output[args.meta_val_way * args.meta_val_shot:]
            test_label = target[args.meta_val_way * args.meta_val_shot:]
            train_out = train_out.reshape(args.meta_val_way, args.meta_val_shot, -1).mean(1)
            train_label = train_label[::args.meta_val_shot]
            prediction = metric_prediction(train_out, test_out, train_label, args.meta_val_metric)
            acc = (prediction == test_label).float().mean()
            top1.update(acc.item())
            if not args.disable_tqdm:
                tqdm_test_loader.set_description('Acc {:.2f}'.format(top1.avg * 100))
    return top1.avg


def train(train_loader, model, criterion, optimizer, epoch, scheduler, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    tqdm_train_loader = warp_tqdm(train_loader)
    for i, (input, target) in enumerate(tqdm_train_loader):
        if args.scheduler == 'cosine':
            scheduler.step(epoch * len(train_loader) + i)
        # measure data loading time
        data_time.update(time.time() - end)

        if args.do_meta_train:
            target = torch.arange(args.meta_train_way)[:, None].repeat(1, args.meta_train_query).reshape(-1).long()
        target = target.cuda(non_blocking=True)

        # compute output
        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            output = model(input)
            if args.do_meta_train:
                output = output.cuda(0)
                shot_proto = output[:args.meta_train_shot * args.meta_train_way]
                query_proto = output[args.meta_train_shot * args.meta_train_way:]
                shot_proto = shot_proto.reshape(args.meta_train_way, args.meta_train_shot, -1).mean(1)
                output = -get_metric(args.meta_train_metric)(shot_proto, query_proto)
            loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        if not args.disable_tqdm:
            tqdm_train_loader.set_description('Acc {:.2f}'.format(top1.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', folder='result/default'):
    torch.save(state, folder + '/' + filename)
    if is_best:
        shutil.copyfile(folder + '/' + filename, folder + '/model_best.pth.tar')

class SmoothCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.):
        super(SmoothCrossEntropy, self).__init__()
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        target_probs = torch.full_like(logits, self.epsilon / (logits.shape[1] - 1))
        target_probs.scatter_(1, labels.unsqueeze(1), 1 - self.epsilon)
        return F.kl_div(torch.log_softmax(logits, 1), target_probs, reduction='batchmean')

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def setup_logger(filepath):
    file_formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger('example')

    file_handle_name = "file"
    if file_handle_name in [h.name for h in logger.handlers]:
        return
    if os.path.dirname(filepath) != '':
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
    file_handle = logging.FileHandler(filename=filepath, mode="a")
    file_handle.set_name(file_handle_name)
    file_handle.setFormatter(file_formatter)
    logger.addHandler(file_handle)
    logger.setLevel(logging.DEBUG)
    return logger


def get_scheduler(batches, optimiter):
    """
    cosine will change learning rate every iteration, others change learning rate every epoch
    :param batches: the number of iterations in each epochs
    :return: scheduler
    """
    SCHEDULER = {'step': StepLR(optimiter, args.lr_stepsize, args.lr_gamma),
                 'multi_step': MultiStepLR(optimiter, milestones=[int(.5 * args.epochs), int(.75 * args.epochs)],
                                           gamma=args.lr_gamma),
                 'cosine': CosineAnnealingLR(optimiter, batches * args.epochs, eta_min=1e-9)}
    return SCHEDULER[args.scheduler]


def get_optimizer(module):
    OPTIMIZER = {'SGD': torch.optim.SGD(module.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
                                        nesterov=args.nesterov),
                 'Adam': torch.optim.Adam(module.parameters(), lr=args.lr)}
    return OPTIMIZER[args.optimizer]


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def extract_feature(train_loader, val_loader, model, tag='last'):
    # return out mean, fcout mean, out feature, fcout features
    save_dir = '{}/{}/{}'.format(args.save_path, tag, args.enlarge)
    if os.path.isfile(save_dir + '/output.plk'):
        data = load_pickle(save_dir + '/output.plk')
        return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    model.eval()
    with torch.no_grad():
        # get training mean
        if not os.path.isfile(save_dir + '/output_mean.plk'):
            out_mean, fc_out_mean = [], []
            for i, (inputs, _) in enumerate(warp_tqdm(train_loader)):
                outputs, fc_outputs = model(inputs, True)
                out_mean.append(outputs.cpu().data.numpy())
                if fc_outputs is not None:
                    fc_out_mean.append(fc_outputs.cpu().data.numpy())
            out_mean = np.concatenate(out_mean, axis=0).mean(0)
            if len(fc_out_mean) > 0:
                fc_out_mean = np.concatenate(fc_out_mean, axis=0).mean(0)
            else:
                fc_out_mean = -1
            save_pickle(save_dir + '/output_mean.plk', [out_mean,fc_out_mean])
        else:
            out_mean, fc_out_mean = load_pickle(save_dir + '/output_mean.plk')

        output_dict = collections.defaultdict(list)
        fc_output_dict = collections.defaultdict(list)
        for i, (inputs, labels) in enumerate(warp_tqdm(val_loader)):
            # compute output
            outputs, fc_outputs = model(inputs, True)
            outputs = outputs.cpu().data.numpy()
            if fc_outputs is not None:
                fc_outputs = fc_outputs.cpu().data.numpy()
            else:
                fc_outputs = [None] * outputs.shape[0]
            for out, fc_out, label in zip(outputs, fc_outputs, labels):
                output_dict[label.item()].append(out)
                fc_output_dict[label.item()].append(fc_out)
        all_info = [out_mean, fc_out_mean, output_dict, fc_output_dict]
        save_pickle(save_dir + '/output.plk', all_info)
        return all_info

def extract_feature_tune(train_loader, val_loader, model, tag='best'):
    # return out mean, fcout mean, out feature, fcout features
    save_dir = '{}/{}/{}'.format(args.save_path, tag, args.enlarge)
    if os.path.isfile(save_dir + '/output_tune.plk'):
        data = load_pickle(save_dir + '/output_tune.plk')
        return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    model.eval()
    with torch.no_grad():
        # get training mean
        if not os.path.isfile(save_dir + '/output_mean.plk'):
            out_mean, fc_out_mean = [], []
            for i, (inputs, _) in enumerate(warp_tqdm(train_loader)):
                outputs, fc_outputs = model(inputs, True)
                out_mean.append(outputs.cpu().data.numpy())
                if fc_outputs is not None:
                    fc_out_mean.append(fc_outputs.cpu().data.numpy())
            out_mean = np.concatenate(out_mean, axis=0).mean(0)
            if len(fc_out_mean) > 0:
                fc_out_mean = np.concatenate(fc_out_mean, axis=0).mean(0)
            else:
                fc_out_mean = -1
            save_pickle(save_dir + '/output_mean.plk', [out_mean,fc_out_mean])
        else:
            out_mean = load_pickle(save_dir + '/output_mean.plk')[0]

        output_dict = collections.defaultdict(list)
        for i, (inputs, labels) in enumerate(warp_tqdm(val_loader)):
            # compute output
            outputs, _ = model(inputs, True)
            outputs = outputs.cpu().data.numpy()
            for out,label in zip(outputs, labels):
                output_dict[label.item()].append(out)
        all_info = [out_mean, output_dict]
        save_pickle(save_dir + '/output_tune.plk', all_info)
        return all_info

def get_dataloader(split, aug=False, shuffle=True, out_name=False, sample=None):

    # sample: iter, way, shot, query
    if aug:
        transform = datasets.with_augment(84, disable_random_resize=args.disable_random_resize, jitter=args.jitter)
    else:
        transform = datasets.without_augment(84, enlarge=args.enlarge)
    sets = datasets.DatasetFolder(args.data, args.split_dir, split, transform, out_name=out_name)
    if sample is not None:
        sampler = datasets.CategoriesSampler(sets.labels, *sample)
        loader = torch.utils.data.DataLoader(sets, batch_sampler=sampler,
                                             num_workers=args.workers, pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(sets, batch_size=args.batch_size, shuffle=shuffle,
                                             num_workers=args.workers, pin_memory=True)
    return loader



def warp_tqdm(data_loader):
    if args.disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm.tqdm(data_loader, total=len(data_loader))
    return tqdm_loader


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def load_checkpoint(model, type='best'):
    if type == 'best':
        checkpoint = torch.load('{}/model_best.pth.tar'.format(args.save_path))
    elif type == 'last':
        checkpoint = torch.load('{}/checkpoint.pth.tar'.format(args.save_path))
    else:
        assert False, 'type should be in [best, or last], but got {}'.format(type)
    model.load_state_dict(checkpoint['state_dict'])


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def meta_evaluate(data, train_mean, shot):
    un_list = []
    l2n_list = []
    cl2n_list = []
    for _ in warp_tqdm(range(args.meta_test_iter)):
        train_data, test_data, train_label, test_label = sample_case(data, shot)
        acc = metric_class_type(train_data, test_data, train_label, test_label, shot, train_mean=train_mean,
                                norm_type='CL2N')
        cl2n_list.append(acc)
        acc = metric_class_type(train_data, test_data, train_label, test_label, shot, train_mean=train_mean,
                                norm_type='L2N')
        l2n_list.append(acc)
        acc = metric_class_type(train_data, test_data, train_label, test_label, shot, train_mean=train_mean,
                                norm_type='UN')
        un_list.append(acc)
    un_mean, un_conf = compute_confidence_interval(un_list)
    l2n_mean, l2n_conf = compute_confidence_interval(l2n_list)
    cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)
    return un_mean, un_conf, l2n_mean, l2n_conf, cl2n_mean, cl2n_conf

def meta_evaluate_tune(data, train_mean, shot):
    cl2n_list = []
    for _ in warp_tqdm(range(args.meta_val_iter)):
        train_data, test_data, train_label, test_label = sample_case(data, shot)
        acc = metric_class_type(train_data, test_data, train_label, test_label, shot, train_mean=train_mean,
                                norm_type='CL2N')
        cl2n_list.append(acc)
    cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)
    return cl2n_mean, cl2n_conf

def tune_lambda(train_loader, model, log):
    val_loader = get_dataloader('val', aug=False, shuffle=False, out_name=False)
    load_checkpoint(model, 'best')

    out_mean, out_dict = extract_feature_tune(train_loader, val_loader, model, tag='best')

    acc_val_list_1 = []
    acc_val_list_5 = []

    lmd_list = [0.1, 0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5]
    for lmd in lmd_list:
        args.lmd = lmd
        accuracy_info_shot1 = meta_evaluate_tune(out_dict, out_mean, 1)
        accuracy_info_shot5 = meta_evaluate_tune(out_dict, out_mean, 5)

        acc_1_val = accuracy_info_shot1[0]
        acc_5_val = accuracy_info_shot5[0]
        acc_val_list_1.append(acc_1_val)
        acc_val_list_5.append(acc_5_val)

        print(
            'validation lmd={:0.2f}: Best\nfeature\tCL2N\n{}\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f}))'.format(args.lmd,
            'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
        log.info(
            'validation lmd={:0.2f}: Best\nfeature\tCL2N\n{}\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f}))'.format(args.lmd,
            'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))

    acc_val_list_1 = np.asarray(acc_val_list_1)
    acc_val_list_5 = np.asarray(acc_val_list_5)
    best_acc_1 = acc_val_list_1.max()
    best_lmd_1 = lmd_list[acc_val_list_1.argmax()]
    best_acc_5 = acc_val_list_5.max()
    best_lmd_5 = lmd_list[acc_val_list_5.argmax()]

    print('Best lambda on validation:\n{:0.2f} with 1 shot acc {:.4f}\n{:0.2f} with 5 shot acc {:.4f}'.format(best_lmd_1, best_acc_1,best_lmd_5, best_acc_5))
    log.info('Best lambda on validation:\n{:0.2f} with 1 shot acc {:.4f}\n{:0.2f} with 5 shot acc {:.4f}'.format(best_lmd_1, best_acc_1,best_lmd_5, best_acc_5))

    return best_lmd_1, best_lmd_5

def lshot_prediction(args, knn, lmd, X, unary, support_label, test_label):

    W = create_affinity(X, knn)
    l = bound_update(args, unary, W, lmd)
    out = np.take(support_label, l)
#     acc, _ = get_accuracy(test_label, out) # Update
    acc = (out == test_label).mean()
    return acc

def metric_class_type(gallery, query, support_label, test_label, shot, train_mean=None, norm_type='CL2N'):
    if norm_type == 'CL2N':
        gallery = gallery - train_mean
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query - train_mean
        query = query / LA.norm(query, 2, 1)[:, None]
    elif norm_type == 'L2N':
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query / LA.norm(query, 2, 1)[:, None]

    if args.proto_rect:
        eta = gallery.mean(0) - query.mean(0) # shift
        query = query + eta[np.newaxis,:]
        query_aug = np.concatenate((gallery, query),axis=0)
        gallery_ = gallery.reshape(args.meta_val_way, shot, gallery.shape[-1]).mean(1)
        gallery_ = torch.from_numpy(gallery_)
        query_aug = torch.from_numpy(query_aug)
        distance = get_metric('cosine')(gallery_, query_aug)
        predict = torch.argmin(distance, dim=1)
        cos_sim = F.cosine_similarity(query_aug[:, None, :], gallery_[None, :, :], dim=2)
        cos_sim = 10 * cos_sim
        W = F.softmax(cos_sim,dim=1)
        gallery_list = [(W[predict==i,i].unsqueeze(1)*query_aug[predict==i]).mean(0,keepdim=True) for i in predict.unique()]
        gallery = torch.cat(gallery_list,dim=0).numpy()
    else:
        gallery = gallery.reshape(args.meta_val_way, shot, gallery.shape[-1]).mean(1)

    support_label = support_label[::shot]
    subtract = gallery[:, None, :] - query
    distance = LA.norm(subtract, 2, axis=-1)
    test_label = np.array(test_label)
    # with LapLacianShot
    if args.lshot and args.lmd!=0:
        knn = args.knn
        lmd = args.lmd
        unary = distance.transpose() ** 2
        acc = lshot_prediction(args, knn, lmd, query, unary, support_label, test_label)
    else:
        idx = np.argpartition(distance, args.num_NN, axis=0)[:args.num_NN]
        nearest_samples = np.take(support_label, idx)
        out = mode(nearest_samples, axis=0)[0]
        out = out.astype(int)
        acc = (out == test_label).mean()
    return acc


def create_affinity(X, knn):
    N, D = X.shape
    # print('Compute Affinity ')
    nbrs = NearestNeighbors(n_neighbors=knn).fit(X)
    dist, knnind = nbrs.kneighbors(X)

    row = np.repeat(range(N), knn - 1)
    col = knnind[:, 1:].flatten()
    data = np.ones(X.shape[0] * (knn - 1))
    W = sparse.csc_matrix((data, (row, col)), shape=(N, N), dtype=np.float)
    return W

def get_accuracy(L1, L2):
    # Since the labels may be different we utilize the Hungarian method to ensure the map of
    # the original ground truth labeling with the returned labels from our laplacian update which is similar to clustering.
    if L1.__len__() != L2.__len__():
        print('size(L1) must == size(L2)')

    Label1 = np.unique(L1)
    nClass1 = Label1.__len__()
    Label2 = np.unique(L2)
    nClass2 = Label2.__len__()

    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i][j] = np.nonzero((L1 == Label1[i]) * (L2 == Label2[j]))[0].__len__()

    c = linear_sum_assignment(-G.T)[1]
    newL2 = np.zeros(L2.__len__())
    for i in range(nClass2):
        for j in np.nonzero(L2 == Label2[i])[0]:
            if len(Label1) > c[i]:
                newL2[j] = Label1[c[i]]

    return accuracy_score(L1, newL2),newL2

def sample_case(ld_dict, shot):
    # Sample meta task
    sample_class = random.sample(list(ld_dict.keys()), args.meta_val_way)
    train_input = []
    test_input = []
    test_label = []
    train_label = []
    for each_class in sample_class:
        total_samples = shot + args.meta_val_query
        if len(ld_dict[each_class]) < total_samples:
            total_samples = len(ld_dict[each_class])

        samples = random.sample(ld_dict[each_class], total_samples)
        train_label += [each_class] * len(samples[:shot])
        test_label += [each_class] * len(samples[shot:])
        train_input += samples[:shot]
        test_input += samples[shot:]
    train_input = np.array(train_input).astype(np.float32)
    test_input = np.array(test_input).astype(np.float32)
    return train_input, test_input, train_label, test_label


def do_extract_and_evaluate(model, log):
    train_loader = get_dataloader('train', aug=False, shuffle=False, out_name=False)
    if args.tune_lmd :
        print('Tuning Lambda')
        best_lmd_1, best_lmd_5 = tune_lambda(train_loader, model, log)
    else:
        best_lmd_1 = best_lmd_5 = args.lmd
    val_loader = get_dataloader('test', aug=False, shuffle=False, out_name=False)
    print(' Proto-rectification = {} in Evaluation'.format(args.proto_rect))
    log.info(' Proto-rectification = {} in Evaluation'.format(args.proto_rect))
    ## With the last model trained on source dataset
    load_checkpoint(model, 'last')
    out_mean, fc_out_mean, out_dict, fc_out_dict = extract_feature(train_loader, val_loader, model, 'last')
    args.lmd = best_lmd_1
    print(' Run with lambda {} for 1 shot'.format(args.lmd))
    log.info(' Run with lambda {} for 1 shot'.format(args.lmd))
    accuracy_info_shot1 = meta_evaluate(out_dict, out_mean, 1)
    args.lmd = best_lmd_5
    print(' Run with lambda {} for 5 shot'.format(args.lmd))
    log.info(' Run with lambda {} for 5 shot'.format(args.lmd))
    accuracy_info_shot5 = meta_evaluate(out_dict, out_mean, 5)
    print(
        'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
            'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
    log.info(
        'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
            'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
    ## With the best model trained on source dataset
    load_checkpoint(model, 'best')
    out_mean, fc_out_mean, out_dict, fc_out_dict = extract_feature(train_loader, val_loader, model, 'best')
    args.lmd = best_lmd_1
    print(' Run with lambda {} for 1 shot'.format(args.lmd))
    log.info(' Run with lambda {} for 1 shot'.format(args.lmd))
    accuracy_info_shot1 = meta_evaluate(out_dict, out_mean, 1)
    args.lmd = best_lmd_5
    print(' Run with lambda {} for 5 shot'.format(args.lmd))
    log.info(' Run with lambda {} for 5 shot'.format(args.lmd))
    accuracy_info_shot5 = meta_evaluate(out_dict, out_mean, 5)
    print(
        'Meta Test: BEST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
            'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
    log.info(
        'Meta Test: BEST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
            'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))


if __name__ == '__main__':
    main()
