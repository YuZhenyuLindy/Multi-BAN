# -*- coding: utf-8 -*
import random
import time
import warnings
import sys
import argparse
import copy
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import os.path as osp
import gc

from network_single import ImageClassifier, DomainDiscriminator
import backbone as BackboneNetwork
from utils import ContinuousDataloader
from transforms import ResizeImage
from lr_scheduler import LrScheduler
from data_list import ImageList
from Loss import *

def get_current_time():
    time_stamp = time.time()
    local_time = time.localtime(time_stamp)
    str_time = time.strftime('%Y-%m-%d_%H-%M-%S', local_time)
    return str_time

def main(args: argparse.Namespace, config):
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
    # cudnn.benchmark = True

    # load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.center_crop:
        train_transform = transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        train_transform = transforms.Compose([
            ResizeImage(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

    val_tranform = transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    data_root = os.path.dirname(args.s_dset_path)
    train_source_dataset = ImageList(data_root, open(args.s_dset_path).readlines(), transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = ImageList(data_root, open(args.t_dset_path).readlines(), transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)

    val_dataset = ImageList(data_root, open(args.t_dset_path).readlines(), transform=val_tranform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    if args.dset == 'domainnet':
        test_dataset = ImageList(data_root, open(args.t_test_path).readlines(), transform=val_tranform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
    else:
        test_loader = val_loader

    train_source_iter = ContinuousDataloader(train_source_loader)
    train_target_iter = ContinuousDataloader(train_target_loader)

    # load model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = BackboneNetwork.__dict__[args.arch](pretrained=True)
    if args.dset == "office":
        num_classes = 31
    elif args.dset == "office-home":
        num_classes = 65
    elif args.dset == "domainnet":
        num_classes = 345
    elif args.dset == "visda2017":
        num_classes = 12
    args.num_classes = num_classes

    if args.dset != "office":
        args.early_stop = True
    stop_count = 0
    classifier = ImageClassifier(backbone, num_classes, args.bottle_neck, args=args).cuda()
    domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).cuda()

    print(classifier)
    config["out_file"].write(classifier.__str__())
    config["out_file"].flush()

    # define optimizer and lr scheduler
    all_parameters = classifier.get_parameters() + domain_discri.get_parameters()
    optimizer = SGD(all_parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    # lr_scheduler = LrScheduler(optimizer, init_lr=args.lr, gamma=0.001, decay_rate=0.75)
    lr_scheduler = LrScheduler(optimizer, init_lr=args.lr, gamma=args.lr_gamma, decay_rate=args.lr_decay)

    # define loss function
    MND_adv = AdversarialLoss_MND(classifier.head, args).cuda()
    domain_adv = DomainAdversarialLoss(domain_discri).cuda()

    # resume from the best checkpoint
    if args.phase != 'train':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(osp.join(args.output_dir, task + "_best.pth"), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        from analysis import collect_feature_output
        import tsne
        import a_distance
        train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.workers, drop_last=False)
        train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.workers, drop_last=False)
        # extract features from both domains
        source_feature, source_output, source_label = collect_feature_output(train_source_loader, classifier, device)
        target_feature, target_output, target_label = collect_feature_output(train_target_loader, classifier, device)
        # plot t-SNE
        tSNE_filename = osp.join(args.output_dir, task + '_TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename, source_color='b',target_color='r')
        print("Saving t-SNE to", tSNE_filename)
        config["out_file"].write("Saving t-SNE to " + tSNE_filename + "\n")
        config["out_file"].flush()

        random.seed(time.time())
        torch.manual_seed(time.time())
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device=device)
        print("A-distance =", A_distance)
        config["out_file"].write("A-distance = {:.6f} \n".format(A_distance))
        config["out_file"].flush()

        # calculate Local A-distance, which is a measure for distribution discrepancy
        Local_A_distance, Local_A_distance_avg = a_distance.local_a_distance(source_feature, target_feature, source_label, target_label, device, num_classes)
        print("Local-A-distance = {:.6f}, Local-A-disntace-avg = {:.6f}".format(Local_A_distance, Local_A_distance_avg))
        config["out_file"].write("Local-A-distance = {:.6f}, Local-A-disntace-avg = {:.6f} \n".format(Local_A_distance, Local_A_distance_avg))
        config["out_file"].flush()

        # calculate mmd loss
        import mmd
        mmd_instance = mmd.MaximumMeanDiscrepancy(num_classes, device=torch.device("cpu"))
        mmd_loss = mmd_instance.mmd_loss(source_feature, target_feature)
        lmmd_loss = mmd_instance.lmmd_loss(source_feature, target_feature, source_label, target_label, is_pseudo_target=False)
        print("mmd_loss = {:.6f}, lmmd_loss = {:.6f}".format(mmd_loss.item(), lmmd_loss.item()))
        config["out_file"].write("mmd_loss = {:.6f}, lmmd_loss = {:.6f} \n".format(mmd_loss.item(), lmmd_loss.item()))
        config["out_file"].flush()
        return

    if args.phase == 'plot':
        from common.utils.analysis import plot_hist_outputs, collect_output, accuracy_by_threshold
        train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=False)
        train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=False)
        # extract outputs from both domains
        output_extractor = classifier.to(device)
        source_output, source_label = collect_output(train_source_loader, output_extractor, device, use_softmax=False)
        target_output, target_label = collect_output(train_target_loader, output_extractor, device, use_softmax=False)
        hist_filename = osp.join(args.output_dir, task + '_Hist.pdf')

        plot_hist_outputs(source_output, source_label, target_output, target_label, hist_filename, use_ground_truth=True, density=True)
        print("Saving hist to", hist_filename)
        config["out_file"].write("Saveing hist to " + hist_filename)
        config["out_file"].flush()
        return

    if args.phase == 'plot_threshold':
        from analysis import plot_hist_outputs, collect_output, accuracy_by_threshold
        train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=False)
        # extract outputs from both domains
        output_extractor = classifier.to(device)
        target_output, target_label = collect_output(train_target_loader, output_extractor, device, use_softmax=True)

        # calculate accuracy by trheshold
        accuracy_threshold = accuracy_by_threshold(target_output, target_label)
        print("Accuracy by threshold")
        print(accuracy_threshold)
        config["out_file"].write("Accuracy by threshold:")
        config["out_file"].write(str(accuracy_threshold))
        config["out_file"].flush()
        return

    if args.phase == 'test':
        # Test for Confusion Matrix
        if args.is_confusion_matrix:
            if args.dset == 'office':
                CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
                           'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
                           'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
                           'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']
            else:#visda-2017
                CLASSES = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
                            'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']
            from analysis import get_confusion_matrix, plot_confusion_matrix
            confusion_matrix = get_confusion_matrix(test_loader, classifier, num_classes)
            confusion_matrix = confusion_matrix.cpu().data.numpy()
            plot_confusion_matrix(confusion_matrix, CLASSES, png_output=args.output_dir, normalize=True)

            # Print Accuracy for Per-Class
            acc_global = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
            acc_per_class = np.diag(confusion_matrix) / np.sum(confusion_matrix, 1)
            display_str = "Accuracy for Global: {}\nAccuracy for Per-Class: {}\n".format(acc_global * 100, ['{:.2f}'.format(i) for i in (acc_per_class * 100).tolist()])
        else:
            acc1 = validate(test_loader, classifier)
            display_str = "acc1 = {:.3f}".format(acc1)

        print(display_str)
        config["out_file"].write(display_str)
        config["out_file"].flush()
        return

    # start training
    best_acc1 = 0.0
    for epoch in range(args.epochs):
        if args.early_stop and stop_count > args.max_stop:
            break

        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, optimizer, domain_adv, MND_adv,
              lr_scheduler, epoch, args) 

        if args.dset == "domainnet":
            if epoch >= 1:
            # if epoch >= 5:
                # evaluate on test set
                acc1 = validate(test_loader, classifier)
                # remember best top1 accuracy and checkpoint
                if acc1 > best_acc1:
                    best_model = copy.deepcopy(classifier.state_dict())
                best_acc1 = max(acc1, best_acc1)
                print("epoch= {:02d},  acc1={:.3f}, best_acc1 = {:.3f}".format(epoch, acc1, best_acc1))
                config["out_file"].write("epoch = {:02d},  acc1 = {:.3f}, best_acc1 = {:.3f}".format(epoch, acc1, best_acc1) + '\n')
                config["out_file"].flush()
        else:
            # evaluate on test set
            acc1 = validate(test_loader, classifier)
            # remember the best top1 accuracy and checkpoint
            if acc1 > best_acc1:
                best_model = copy.deepcopy(classifier.state_dict())
                if args.is_save:
                    torch.save(best_model, osp.join(args.output_dir, config["prefix_out_file"] + "_" + task + "_best.pth"))
                stop_count = 0
            stop_count = stop_count + 1

            best_acc1 = max(acc1, best_acc1)
            print("epoch = {:02d},  acc1={:.3f}, best_acc1 = {:.3f}".format(epoch, acc1, best_acc1))
            config["out_file"].write("epoch = {:02d},  acc1= {:.3f}, best_acc1 = {:.3f}".format(epoch, acc1, best_acc1) + '\n')
            config["out_file"].flush()

        if args.is_save:
            torch.save(classifier.state_dict(), osp.join(args.output_dir, config["prefix_out_file"] +  "_" + task + "_latest.pth"))

    print("best_acc1 = {:.3f}".format(best_acc1))
    config["out_file"].write("best_acc1 = {:.3f}".format(best_acc1) + '\n')
    config["out_file"].flush()

def train(train_source_iter: ContinuousDataloader, train_target_iter: ContinuousDataloader, model: ImageClassifier,
        optimizer: SGD, domain_adv, MND_adv, lr_scheduler: LrScheduler, epoch: int, args: argparse.Namespace):
    # switch to train mode
    model.train()
    domain_adv.train()
    MND_adv.train()

    max_iters = args.iters_per_epoch * args.epochs
    for i in range(args.iters_per_epoch):
        current_iter = i + args.iters_per_epoch * epoch
        rho = current_iter / max_iters
        lr_scheduler.step()

        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)

        x_s = x_s.cuda()
        x_t = x_t.cuda()
        labels_s = labels_s.cuda()

        # get features and logit outputs
        x = torch.cat((x_s, x_t), dim=0)
        y,f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)
        if args.feature_normal:
            y_s, y_t  = y_s / args.feature_temp, y_t / args.feature_temp

        # compute loss
        cls_loss = F.cross_entropy(y_s, labels_s)
        loss_mnd = MND_adv(f, labels_s, args)
        total_loss = cls_loss - args.mnd_tradeoff * loss_mnd

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # print training log
        if i % args.print_freq == 0:
            print("Epoch: [{:02d}][{}/{}]	total_loss:{:.3f}	cls_loss:{:.3f}	  mnd_loss:{:.3f}".format(\
                epoch, i, args.iters_per_epoch, total_loss, cls_loss, loss_mnd))
    return

def validate(val_loader: DataLoader, model: ImageClassifier) -> float:
    # switch to evaluate mode
    model.eval()
    start_test = True
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            # get logit outputs
            output, _ = model(images)
            if start_test:
                all_output = output.float()
                all_label = target.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, output.float()), 0)
                all_label = torch.cat((all_label, target.float()), 0)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        accuracy = accuracy * 100.0
        print(' accuracy:{:.3f}'.format(accuracy))
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-batch Nuclear-norm Adversarial Network for Unsupervised Domain Adaptation')
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'resnet101'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='3', help="device id to run")
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'office-home', 'domainnet', 'visda2017'], help="The dataset used")
    parser.add_argument('--s_dset_path', type=str, default='/data1/TL/data/office31/amazon_list.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='/data1/TL/data/office31/webcam_list.txt', help="The target dataset path list")
    parser.add_argument('--t_test_path', type=str, default='/data1/TL/data/office31/webcam_list.txt', help="The target test dataset path list")
    parser.add_argument('--output_dir', type=str, default='log/MBAN/office31', help="output directory of logs")
    parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--iters-per-epoch', default=500, type=int, help='Number of iterations per epoch')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--batch-size', default=36, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', default=1e-3, type=float, metavar='W', help='weight decay (default: 1e-3)', dest='weight_decay')
    parser.add_argument('--center_crop', default=False, action='store_true')
    parser.add_argument('--bottle-neck', default=256, type=int, help='dims for bottle neck layer. ')
    parser.add_argument('--seed', default=2, type=int, help='seed for initializing training. ')
    parser.add_argument('--temp', type=float, default=10.0, help="temperature scaling parameter")
    parser.add_argument('--feature_normal', type=bool, default=False, help="whether use normalization for feature")
    parser.add_argument('--output_temp', type=float, default=1.0, help="temperature for output")
    parser.add_argument('--logit_temp', type=float, default=2.5, help="temperature for target logit")
    parser.add_argument('--queue-K', type=int, default=3, help="the number of cached multi-batch or sample of per-classes)")
    parser.add_argument('--cache-feature', action='store_true', help="whether cache feature")
    parser.add_argument('--mnd_tradeoff', type=float, default=0.1, help="hyper-parameter:loss_entropy")
    parser.add_argument('--weight_type', type=int, default=0, help="the type of the estimated weight")
    parser.add_argument('--early_stop', type=bool, default=False, help="whether early stop")
    parser.add_argument('--max_stop', type=int, default=10, help="max epoch for early stop")
    parser.add_argument('--is_confusion_matrix', type=bool, default=False, help="whether use consfuison matrix")
    parser.add_argument('--is_save', action='store_true', help="whether save model")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis','plot', 'plot_threshold'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()

    config = {}
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
    task = args.s_dset_path.split('/')[-1].split('.')[0].split('_')[0] + "-" + \
           args.t_dset_path.split('/')[-1].split('.')[0].split('_')[0]
    config["prefix_out_file"] = get_current_time()
    config["out_file"] = open(osp.join(args.output_dir, config["prefix_out_file"] + "_" + task + "-" + args.phase + "_log.txt"), "w")

    config["out_file"].write("train_MBAN.py\n")
    import PIL
    config["out_file"].write("PIL version: {}\ntorch version: {}\ntorchvision version: {}\n".format(PIL.__version__, torch.__version__, torchvision.__version__))
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
        config["out_file"].write(str("{} = {}".format(arg, getattr(args, arg))) + "\n")
    config["out_file"].flush()
    main(args, config)
