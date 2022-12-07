import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib

import models.my_models
from Wavelets import wpt_transform

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import os
import pprint
import argparse
import time
import json
import pprint

# dataset and models
from dataset import ChexpertSmall, extract_patient_ids
from torchvision.models import densenet121, resnet152

parser = argparse.ArgumentParser()
# action
parser.add_argument('--load_config', type=str, help='Path to config.json file to load args from.')
parser.add_argument('--train', action='store_true', help='Train model.')
parser.add_argument('--evaluate_single_model', action='store_true', help='Evaluate a single model.')
parser.add_argument('--evaluate_ensemble', action='store_true',
                    help='Evaluate an ensemble (given a checkpoints tracker of saved model checkpoints).')
parser.add_argument('--visualize', action='store_true', help='Visualize Grad-CAM.')
parser.add_argument('--plot_roc', action='store_true', help='Filename for metrics json file to plot ROC.')
parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
parser.add_argument('--cuda', type=int, help='Which cuda device to use.')
# paths
parser.add_argument('--data_path', default='',
                    help='Location of train/valid datasets directory or path to test csv file.')
parser.add_argument('--output_dir', help='Path to experiment output, config, checkpoints, etc.')
parser.add_argument('--restore', type=str,
                    help='Path to a single model checkpoint to restore or folder of checkpoints to ensemble.')
# model architecture
parser.add_argument('--model', default='densenet121',
                    help='What model architecture to use. (densenet121, resnet152, wpt_resnet152)')
# data params
parser.add_argument('--mini_data', type=int, help='Truncate dataset to this number of examples.')
parser.add_argument('--f_key', type=str, help='pass filter key parameters e.g. Path, Frontal/Lateral')
parser.add_argument('--f_value', type=str, help='pass filter value parameters e.g. Frontal Lateral')
parser.add_argument('--resize', type=int, help='Size of minimum edge to which to resize images.')
parser.add_argument('--frac', type=int, help='fraction of the data to use (e.g. use 80% of total train)')
parser.add_argument('--ext', default='img', help='What data type extension to use [img, qwp]')
# wpt parameters
parser.add_argument('--wpt_d', type=int, default=3, help='set wpt d level, default 3')
parser.add_argument('--wpt_expand', type=int, help='set wpt expanded dimension, default None')
parser.add_argument('--wpt_energy_sort', action='store_true', help='sort by wpt energy ')
parser.add_argument('--wpt_nfreq', type=int, help='n freq to take into account')
parser.add_argument('--wpt_norm', action='store_true', help='set if normalized data needed')
parser.add_argument('--wpt_use_originals', action='store_true', help='set if want to stack wpt on original image')

# training params
parser.add_argument('--input_ch', type=int, default=64, help='number of input channels to network.')
parser.add_argument('--pretrained', action='store_true',
                    help='Use ImageNet pretrained model and normalize data mean and std.')
parser.add_argument('--batch_size', type=int, default=16, help='Dataloaders batch size.')
parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--lr_warmup_steps', type=float, default=0,
                    help='Linear warmup of the learning rate for lr_warmup_steps number of steps.')
parser.add_argument('--lr_decay_factor', type=float, default=0.97,
                    help='Decay factor if exponential learning rate decay scheduler.')
parser.add_argument('--step', type=int, default=0, help='Current step of training (number of minibatches processed).')
parser.add_argument('--log_interval', type=int, default=50, help='Interval of num batches to show loss statistics.')
parser.add_argument('--eval_interval', type=int, default=300,
                    help='Interval of num epochs to evaluate, checkpoint, and save samples.')
parser.add_argument('--num_workers', type=int, default=0, help='define number of workers to use for training')


# --------------------
# Data IO
# --------------------

def fetch_dataloader(args, mode):
    assert mode in ['train', 'valid', 'vis']
    transformations = [
        T.Resize(args.resize) if args.resize else T.Lambda(lambda x: x),
        T.CenterCrop(320 if not args.resize else args.resize),
        lambda x: torch.from_numpy(np.array(x, copy=True)).float().div(255).unsqueeze(0),  # tensor in [0,1]
        T.Normalize(mean=[0.5330], std=[0.0349]),  # whiten with dataset mean and std
    ]

    if args.ext == "img":
        transformations.append(lambda x: x.expand(3, -1, -1))  # expand to 3 channels
    elif args.ext == 'qwp':
        # transformations.append(T.Resize(256))  # resize to power of 2 dimension  (H, W) before qWPT
        transformations.append(wpt_transform.qWPT(
            DeTr=args.wpt_d,
            expand=args.wpt_expand,
            norm=args.wpt_norm,
            energy_sort=args.wpt_energy_sort,
            nfreq=args.wpt_nfreq,
            use_originals=args.wpt_use_originals,
        ))  # preform qWPT

    transforms = T.Compose(transformations)
    dataset = ChexpertSmall(args.data_path, mode, transforms, data_filter=args.filter, mini_data=args.mini_data,
                            ext=args.ext)
    return DataLoader(dataset, args.batch_size, shuffle=(mode == 'train'), pin_memory=(args.device.type == 'cuda'),
                      num_workers=0 if mode == 'valid' else args.num_workers)  # since evaluating the valid_dataloader
    # is called inside the train_dataloader loop, 0 workers for valid_dataloader avoids
    # forking (cf torch dataloader docs); else memory sharing gets clunky


def save_json(data, filename, args):
    with open(os.path.join(args.output_dir, filename + '.json'), 'w') as f:
        json.dump(data, f, indent=4)


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def save_checkpoint(checkpoint, optim_checkpoint, sched_checkpoint, args, max_records=10):
    """ save model and optimizer checkpoint along with csv tracker
    of last `max_records` best number of checkpoints as sorted by avg auc """
    # 1. save latest
    torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint_latest.pt'))
    torch.save(optim_checkpoint, os.path.join(args.output_dir, 'optim_checkpoint_latest.pt'))
    if sched_checkpoint: torch.save(sched_checkpoint, os.path.join(args.output_dir, 'sched_checkpoint_latest.pt'))

    # 2. save the last `max_records` number of checkpoints as sorted by avg auc
    tracker_path = os.path.join(args.output_dir, 'checkpoints_tracker.csv')
    tracker_header = ' '.join(['CheckpointId', 'Step', 'Loss', 'AvgAUC'])

    # 2a. load checkpoint stats from file
    old_data = None  # init and overwrite from records
    file_id = 0  # init and overwrite from records
    lowest_auc = float('-inf')  # init and overwrite from records
    if os.path.exists(tracker_path):
        old_data = np.atleast_2d(np.loadtxt(tracker_path, skiprows=1))
        file_id = len(old_data)
        if len(old_data) == max_records:  # remove the lowest-roc record and add new checkpoint record under its file-id
            lowest_auc_idx = old_data[:, 3].argmin()
            lowest_auc = old_data[lowest_auc_idx, 3]
            file_id = int(old_data[lowest_auc_idx, 0])
            old_data = np.delete(old_data, lowest_auc_idx, 0)

    # 2b. update tracking data and sort by descending avg auc
    data = np.atleast_2d([file_id, args.step, checkpoint['eval_loss'], checkpoint['avg_auc']])
    if old_data is not None: data = np.vstack([old_data, data])
    data = data[data.argsort(0)[:, 3][::-1]]  # sort descending by AvgAUC column

    # 2c. save tracker and checkpoint if better than what is already saved
    if checkpoint['avg_auc'] > lowest_auc:
        np.savetxt(tracker_path, data, delimiter=' ', header=tracker_header)
        torch.save(checkpoint, os.path.join(args.output_dir, 'best_checkpoints', 'checkpoint_{}.pt'.format(file_id)))


# --------------------
# Evaluation metrics
# --------------------
def add_pr_curve_tensorboard(class_index, test_probs, test_label, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_truth = test_label == class_index
    tensorboard_probs = test_probs

    writer.add_pr_curve(ChexpertSmall.attr_names[class_index],
                        tensorboard_truth,
                        tensorboard_probs,
                        global_step=global_step)


def compute_metrics(outputs, targets, losses):
    # add accuracy_score, average_precision_score, top_k_accuracy_score
    n_classes = outputs.shape[1]
    # getting true predicated as [0, 1]
    _outputs = torch.round(torch.sigmoid(outputs))
    N = outputs.shape[0]
    accuracy = ((_outputs == targets).sum() / (N * n_classes) * 100).item()
    fpr, tpr, aucs, precision, recall, acc_score, avg_p_score = {}, {}, {}, {}, {}, {}, {}
    for i in range(n_classes):
        acc_score[i] = accuracy_score(targets[:, i], _outputs[:, i])
        fpr[i], tpr[i], _ = roc_curve(targets[:, i], outputs[:, i])
        aucs[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(targets[:, i], outputs[:, i])
        fpr[i], tpr[i], precision[i], recall[i] = fpr[i].tolist(), tpr[i].tolist(), precision[i].tolist(), recall[
            i].tolist()
        # add_pr_curve_tensorboard(i, _outputs[:, i], targets[:, i], global_step=args.step)
    mean_auc = sum(aucs.values()) / len(aucs) * 100
    new_mean_auc = np.nanmean(list(aucs.values())) * 100

    metrics = {
        'mean_auc': mean_auc,
        'new_mean_auc': new_mean_auc,
        'class_acc': acc_score,
        'accuracy': accuracy,
        'aucs': aucs,
        'loss': dict(enumerate(losses.mean(0).tolist())),
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall
    }

    return metrics

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(10, 10))
    for idx in np.arange(images.shape[0]):
        ax = fig.add_subplot((images.shape[0] // 4), 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            ChexpertSmall.attr_names[preds[idx]],
            probs[idx] * 100.0,
            ChexpertSmall.attr_names[int(labels[idx][preds[idx]].item())]),
                    color=("green" if preds[idx] == labels[idx][preds[idx]].item() else "red"))
    fig.tight_layout()
    return fig
# --------------------
# Train and evaluate
# --------------------

def train_epoch(model, train_dataloader, valid_dataloader, loss_fn, optimizer, scheduler, writer, epoch, args):
    model.train()
    # images, _, _ = next(iter(train_dataloader))
    # # grid = torchvision.utils.make_grid(images)
    # # grid = torchvision.utils.make_grid(images)
    # # writer.add_image("images", grid)
    # writer.add_graph(model, images)
    # end = time.time()
    with tqdm(total=len(train_dataloader),
              desc=f'Step at start {args.step}; Training epoch {epoch + 1}/{args.n_epochs}') as pbar:
        for x, target, idxs in train_dataloader:
            args.step += 1
            # data_time.update(time.time() - end)
            out = model(x.to(args.device))
            loss = loss_fn(out, target.to(args.device)).sum(1).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if scheduler and args.step >= args.lr_warmup_steps: scheduler.step()

            # record
            if args.step % args.log_interval == 0:
                writer.add_scalar('loss/train_loss', loss.item(), args.step)

            pbar.set_postfix(
                loss='{:.4f}'.format(loss.item())
            )
            pbar.update()
            # evaluate and save on eval_interval
            if args.step % args.eval_interval == 0:
                with torch.no_grad():
                    model.eval()

                    eval_metrics = evaluate_single_model(model, valid_dataloader, loss_fn, args)
                    # writer.add_figure('predictions vs. actuals',
                    #                   plot_classes_preds(net, inputs, labels),
                    #                   global_step=args.step)
                    writer.add_scalar('loss/eval_loss', np.sum(list(eval_metrics['loss'].values())), args.step)
                    for k, v in eval_metrics['aucs'].items():
                        writer.add_scalar('auc/eval_auc_class_{}'.format(k), v, args.step)
                    for k, v in eval_metrics['class_acc'].items():
                        writer.add_scalar('accuracy/eval_acc_class_{}'.format(k), v, args.step)
                    writer.add_scalar('accuracy/accuracy', eval_metrics['accuracy'], args.step)
                    writer.add_figure('predictions vs. labels',
                                      plot_classes_preds(model, x.to(args.device), target.to(args.device)),
                                      global_step=args.step)
                    # save model
                    save_checkpoint(checkpoint={'global_step': args.step,
                                                'eval_loss': np.sum(list(eval_metrics['loss'].values())),
                                                'avg_auc': np.nanmean(list(eval_metrics['aucs'].values())),
                                                'avg_acc': np.nanmean(list(eval_metrics['class_acc'].values())),
                                                'accuracy': eval_metrics['accuracy'],
                                                'state_dict': model.state_dict()},
                                    optim_checkpoint=optimizer.state_dict(),
                                    sched_checkpoint=scheduler.state_dict() if scheduler else None,
                                    args=args)
                    pbar.set_postfix(
                        loss='{:.4f}'.format(loss.item()),
                        eval_loss='{:.4f}'.format(np.sum(list(eval_metrics['loss'].values()))),
                        acc='{:.4f}'.format(eval_metrics['accuracy']),
                        avg_auc='{:.4f}'.format(np.nanmean(list(eval_metrics['aucs'].values()))),
                    )
                    pbar.update()
                    # switch back to train mode
                    model.train()


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, args):
    model.eval()

    targets, outputs, losses = [], [], []
    for x, target, idxs in dataloader:
        out = model(x.to(args.device))
        loss = loss_fn(out, target.to(args.device))

        outputs += [out.cpu()]
        targets += [target]
        losses += [loss.cpu()]

    return torch.cat(outputs), torch.cat(targets), torch.cat(losses)


def evaluate_single_model(model, dataloader, loss_fn, args):
    outputs, targets, losses = evaluate(model, dataloader, loss_fn, args)
    return compute_metrics(outputs, targets, losses)


def evaluate_ensemble(model, dataloader, loss_fn, args):
    checkpoints = [c for c in os.listdir(args.restore) if c.startswith('checkpoint') and c.endswith('.pt')]
    print('Running ensemble prediction using {} checkpoints.'.format(len(checkpoints)))
    outputs, losses = [], []
    for checkpoint in checkpoints:
        # load weights
        model_checkpoint = torch.load(os.path.join(args.restore, checkpoint), map_location=args.device)
        model.load_state_dict(model_checkpoint['state_dict'])
        del model_checkpoint
        # evaluate
        outputs_, targets, losses_ = evaluate(model, dataloader, loss_fn, args)
        outputs += [outputs_]
        losses += [losses_]

    # take mean over checkpoints
    outputs = torch.stack(outputs, dim=2).mean(2)
    losses = torch.stack(losses, dim=2).mean(2)

    return compute_metrics(outputs, targets, losses)


def train_and_evaluate(model, train_dataloader, valid_dataloader, loss_fn, optimizer, scheduler, writer, args):
    for epoch in range(args.n_epochs):
        train_epoch(model, train_dataloader, valid_dataloader, loss_fn, optimizer, scheduler, writer, epoch, args)

        # evaluate
        print('Evaluating...', end='\r')
        eval_metrics = evaluate_single_model(model, valid_dataloader, loss_fn, args)
        print('Evaluate metrics @ step {}:'.format(args.step))
        print('AUC:\n', pprint.pformat(eval_metrics['aucs']))
        print('Mean AUC:\n', pprint.pformat(eval_metrics['mean_auc']))
        print('Accuracy:\n', pprint.pformat(eval_metrics['accuracy']))
        print('Class Accuracy:\n', pprint.pformat(eval_metrics['class_acc']))
        print('Loss:\n', pprint.pformat(eval_metrics['loss']))
        writer.add_scalar('loss/eval_loss', np.sum(list(eval_metrics['loss'].values())), args.step)
        for k, v in eval_metrics['aucs'].items():
            writer.add_scalar('auc/eval_auc_class_{}'.format(k), v, args.step)
        for k, v in eval_metrics['class_acc'].items():
            writer.add_scalar('accuracy/eval_acc_class_{}'.format(k), v, args.step)
        writer.add_scalar('accuracy/accuracy', eval_metrics['accuracy'], args.step)
        # save eval metrics
        save_json(eval_metrics, 'eval_results_step_{}'.format(args.step), args)


# --------------------
# Visualization
# --------------------

@torch.enable_grad()
def grad_cam(model, x, hooks, cls_idx=None):
    """ cf CheXpert: Test Results / Visualization; visualize final conv layer, using grads of final linear layer as weights,
    and performing a weighted sum of the final feature maps using those weights.
    cf Grad-CAM https://arxiv.org/pdf/1610.02391.pdf """

    model.eval()
    model.zero_grad()

    # register backward hooks
    conv_features, linear_grad = [], []
    forward_handle = hooks['forward'].register_forward_hook(
        lambda module, in_tensor, out_tensor: conv_features.append(out_tensor))
    backward_handle = hooks['backward'].register_backward_hook(
        lambda module, grad_input, grad_output: linear_grad.append(grad_input))

    # run model forward and create a one hot output for the given cls_idx or max class
    outputs = model(x)
    if not cls_idx: cls_idx = outputs.argmax(1)
    one_hot = F.one_hot(cls_idx, outputs.shape[1]).float().requires_grad_(True)

    # run model backward
    one_hot.mul(outputs).sum().backward()

    # compute weights; cf. Grad-CAM eq 1 -- gradients flowing back are global-avg-pooled to obtain the neuron importance weights
    weights = linear_grad[0][2].mean(1).view(1, -1, 1, 1)
    # compute weighted combination of forward activation maps; cf Grad-CAM eq 2; linear combination over channels
    cam = F.relu(torch.sum(weights * conv_features[0], dim=1, keepdim=True))

    # normalize each image in the minibatch to [0,1] and upscale to input image size
    cam = cam.clone()  # avoid modifying tensor in-place

    def norm_ip(t, min, max):
        t.clamp_(min=min, max=max)
        t.add_(-min).div_(max - min + 1e-5)

    # for t in cam:  # loop over mini-batch dim
    # for i in range(cam.size(0)): # loop over mini-batch dimension
    for idx, _ in enumerate([jj for jj in cam]):
        t = cam[idx]
        # t = cam[i]
        norm_ip(t, float(t.min()), float(t.max()))

    cam = F.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=True)

    # cleanup
    forward_handle.remove()
    backward_handle.remove()
    model.zero_grad()

    return cam


def visualize(model, dataloader, grad_cam_hooks, args):
    attr_names = dataloader.dataset.attr_names

    # 1. run through model to compute logits and grad-cam
    imgs, labels, scores, masks, idxs = [], [], [], [], []
    for x, target, idx in dataloader:
        imgs += [x]
        labels += [target]
        idxs += idx.tolist()
        x = x.to(args.device)
        scores += [model(x).cpu()]
        masks += [grad_cam(model, x, grad_cam_hooks).cpu()]
    imgs, labels, scores, masks = torch.cat(imgs), torch.cat(labels), torch.cat(scores), torch.cat(masks)

    # 2. renormalize images and convert everything to numpy for matplotlib
    imgs.mul_(0.0349).add_(0.5330)
    imgs = imgs.permute(0, 2, 3, 1).data.numpy()
    labels = labels.data.numpy()
    patient_ids = extract_patient_ids(dataloader.dataset, idxs)
    masks = masks.permute(0, 2, 3, 1).data.numpy()
    probs = scores.sigmoid().data.numpy()

    # 3. make column grid of [model probs table, original image, grad-cam image] for each attr + other categories
    for attr, vis_idxs in zip(dataloader.dataset.vis_attrs, dataloader.dataset.vis_idxs):
        fig, axs = plt.subplots(3, 3, figsize=(4 * imgs.shape[1] / 100, 3.3 * imgs.shape[2] / 100), dpi=100,
                                frameon=False)
        fig.suptitle(attr)
        for i, idx in enumerate(vis_idxs):
            offset = idxs.index(idx)
            visualize_one(model, imgs[offset], masks[offset], labels[offset], patient_ids[offset], probs[offset],
                          attr_names, axs[i])

        filename = 'vis_{}_step_{}.png'.format(attr.replace(' ', '_'), args.step)
        plt.savefig(os.path.join(args.output_dir, 'vis', filename), dpi=100)
        plt.close()


def visualize_one(model, img, mask, label, patient_id, prob, attr_names, axs):
    """ display [table of model vs ground truth probs | original image | grad-cam mask image] in a given suplot axs """
    # sort data by prob high to low
    sort_idxs = prob.argsort()[::-1]
    label = label[sort_idxs]
    prob = prob[sort_idxs]
    names = [attr_names[i] for i in sort_idxs]
    # 1. left -- show table of ground truth and predictions, sorted by pred prob high to low
    axs[0].set_title(patient_id)
    data = np.stack([label, prob.round(3)]).T
    axs[0].table(cellText=data, rowLabels=names, colLabels=['Ground truth', 'Pred. prob'],
                 rowColours=plt.cm.Greens(0.5 * label),
                 cellColours=plt.cm.Greens(0.5 * data), cellLoc='center', loc='center')
    axs[0].axis('tight')
    # 2. middle -- show original image
    axs[1].set_title('Original image', fontsize=10)
    axs[1].imshow(img.squeeze(), cmap='gray')
    # 3. right -- show heatmap over original image with predictions
    axs[2].set_title('Top class activation \n{}: {:.4f}'.format(names[0], prob[0]), fontsize=10)
    axs[2].imshow(img.squeeze(), cmap='gray')
    axs[2].imshow(mask.squeeze(), cmap='jet', alpha=0.5)

    for ax in axs: ax.axis('off')


def plot_roc(metrics, args, filename, labels=ChexpertSmall.attr_names):
    fig, axs = plt.subplots(2, len(labels), figsize=(24, 12))

    for i, (fpr, tpr, aucs, precision, recall, label) in enumerate(zip(metrics['fpr'].values(), metrics['tpr'].values(),
                                                                       metrics['aucs'].values(),
                                                                       metrics['precision'].values(),
                                                                       metrics['recall'].values(), labels)):
        # top row -- ROC
        axs[0, i].plot(fpr, tpr, label='AUC = %0.2f' % aucs)
        axs[0, i].plot([0, 1], [0, 1], 'k--')  # diagonal margin
        axs[0, i].set_xlabel('False Positive Rate')
        # bottom row - Precision-Recall
        axs[1, i].step(recall, precision, where='post')
        axs[1, i].set_xlabel('Recall')
        # format
        axs[0, i].set_title(label)
        axs[0, i].legend(loc="lower right")

    plt.suptitle(filename)
    axs[0, 0].set_ylabel('True Positive Rate')
    axs[1, 0].set_ylabel('Precision')

    for ax in axs.flatten():
        ax.set_xlim([0.0, 1.05])
        ax.set_ylim([0.0, 1.05])
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'plots', filename + '.png'), pad_inches=0.)
    plt.close()


# --------------------
# Main
# --------------------

if __name__ == '__main__':
    args = parser.parse_args()
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(args.__dict__)
    if args.f_key is None or args.f_value is None:
        my_filter = {}
    else:
        my_filter = {args.f_key: args.f_value}
    args.filter = my_filter

    # overwrite args from config
    if args.load_config:
        args.__dict__.update(load_json(args.load_config))

    # set up output folder
    if not args.output_dir:
        if args.restore:
            raise RuntimeError('Must specify `output_dir` argument')
        args.output_dir: args.output_dir = os.path.join('results', time.strftime('%Y-%m-%d_%H-%M', time.gmtime()))

    # make new folders if they don't exist
    # writer = SummaryWriter(logdir=args.output_dir)
    writer = SummaryWriter()
    # creates output_dir
    if not os.path.exists(os.path.join(args.output_dir, 'vis')):
        os.makedirs(os.path.join(args.output_dir, 'vis'))
    if not os.path.exists(os.path.join(args.output_dir, 'plots')):
        os.makedirs(os.path.join(args.output_dir, 'plots'))
    if not os.path.exists(os.path.join(args.output_dir, 'best_checkpoints')):
        os.makedirs(os.path.join(args.output_dir, 'best_checkpoints'))

    # save config
    if not os.path.exists(os.path.join(args.output_dir, 'config.json')):
        save_json(args.__dict__, 'config', args)
    with open('config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    writer.add_text('config', str(args.__dict__))

    args.device = torch.device(
        'cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu')
    print(f"using {args.device} as device")

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # load model
    n_classes = len(ChexpertSmall.attr_names)
    if args.model == 'densenet121':
        # if args.pretrained:
        #     model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1).to(args.device)
        # else:
        model = densenet121().to(args.device)
        # 1. replace output layer with chexpert number of classes (pretrained loads ImageNet n_classes)
        model.classifier = nn.Linear(model.classifier.in_features, out_features=n_classes).to(args.device)
        # 2. init output layer with default torchvision init
        nn.init.constant_(model.classifier.bias, 0)
        # 3. store locations of forward and backward hooks for grad-cam
        grad_cam_hooks = {'forward': model.features.norm5, 'backward': model.classifier}
        # 4. init optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
    elif args.model == 'wptdensenet121':
        model = models.my_models.wptdensenet121(args.input_ch).to(args.device)
        # 1. replace output layer with chexpert number of classes (pretrained loads ImageNet n_classes)
        model.classifier = nn.Linear(model.classifier.in_features, out_features=n_classes).to(args.device)
        # 2. init output layer with default torchvision init
        nn.init.constant_(model.classifier.bias, 0)
        # 3. store locations of forward and backward hooks for grad-cam
        grad_cam_hooks = {'forward': model.features.norm5, 'backward': model.classifier}
        # 4. init optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
    elif args.model == 'resnet152':
        # if args.pretrained:
        #     model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2).to(args.device)
        # else:
        model = models.my_models.resnet_152().to(args.device)
        model.fc = nn.Linear(model.fc.in_features, out_features=n_classes).to(args.device)
        grad_cam_hooks = {'forward': model.layer4, 'backward': model.fc}
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
    elif args.model == 'dual':
        model = models.my_models.DualResNet().to(args.device)
        model.final_fc1 = nn.Linear(model.final_fc1.in_features, out_features=n_classes).to(args.device)
        grad_cam_hooks = {'forward': model.layer4, 'backward': model.fc}
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
    elif args.model == 'wpt_resnet152':
        model = models.my_models.wpt_resnet_152(args.input_ch).to(args.device)
        model.fc = nn.Linear(model.fc.in_features, out_features=n_classes).to(args.device)
        model = model.double()
        grad_cam_hooks = {'forward': model.layer4, 'backward': model.fc}
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
    else:
        raise RuntimeError('Model architecture not supported.')

    if args.restore and os.path.isfile(
            args.restore):  # restore from single file, else ensemble is handled by evaluate_ensemble
        print('Restoring model weights from {}'.format(args.restore))
        model_checkpoint = torch.load(args.restore, map_location=args.device)
        model.load_state_dict(model_checkpoint['state_dict'])
        args.step = model_checkpoint['global_step']
        del model_checkpoint
        # if training, load optimizer and scheduler too
        if args.train:
            print('Restoring optimizer.')
            optim_checkpoint_path = os.path.join(os.path.dirname(args.restore),
                                                 'optim_' + os.path.basename(args.restore))
            optimizer.load_state_dict(torch.load(optim_checkpoint_path, map_location=args.device))
            if scheduler:
                print('Restoring scheduler.')
                sched_checkpoint_path = os.path.join(os.path.dirname(args.restore),
                                                     'sched_' + os.path.basename(args.restore))
                scheduler.load_state_dict(torch.load(sched_checkpoint_path, map_location=args.device))

    # load data
    if args.restore:
        # load pretrained flag from config -- in case forgotten e.g. in post-training evaluation
        # (images still need to be normalized if training started on an imagenet pretrained model)
        args.pretrained = load_json(os.path.join(args.output_dir, 'config.json'))['pretrained']
    train_dataloader = fetch_dataloader(args, mode='train')
    valid_dataloader = fetch_dataloader(args, mode='valid')
    vis_dataloader = fetch_dataloader(args, mode='vis')

    # setup loss function for train and eval
    loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(args.device)

    print('Loaded {} (number of parameters: {:,}; weights trained to step {})'.format(
        model._get_name(), sum(p.numel() for p in model.parameters()), args.step))
    print('Train data length: ', len(train_dataloader.dataset))
    print('Valid data length: ', len(valid_dataloader.dataset))
    print('Vis data subset: ', len(vis_dataloader.dataset))

    if args.train:
        train_and_evaluate(model, train_dataloader, valid_dataloader, loss_fn, optimizer, scheduler, writer, args)

    if args.evaluate_single_model:
        eval_metrics = evaluate_single_model(model, valid_dataloader, loss_fn, args)
        print('Evaluate metrics -- \n\t restore: {} \n\t step: {}:'.format(args.restore, args.step))
        print('AUC:\n', pprint.pformat(eval_metrics['aucs']))
        print('Mean AUC:\n', pprint.pformat(eval_metrics['mean_auc']))
        print('Accuracy:\n', pprint.pformat(eval_metrics['accuracy']))
        print('Class Accuracy:\n', pprint.pformat(eval_metrics['class_acc']))
        print('Loss:\n', pprint.pformat(eval_metrics['loss']))
        save_json(eval_metrics, 'eval_results_step_{}'.format(args.step), args)

    if args.evaluate_ensemble:
        assert os.path.isdir(args.restore), 'Restore argument must be directory with saved checkpoints'
        eval_metrics = evaluate_ensemble(model, valid_dataloader, loss_fn, args)
        print('Evaluate ensemble metrics -- \n\t checkpoints path {}:'.format(args.restore))
        print('AUC:\n', pprint.pformat(eval_metrics['aucs']))
        print('Mean AUC:\n', pprint.pformat(eval_metrics['mean_auc']))
        print('Accuracy:\n', pprint.pformat(eval_metrics['accuracy']))
        print('Class Accuracy:\n', pprint.pformat(eval_metrics['class_acc']))
        print('Loss:\n', pprint.pformat(eval_metrics['loss']))
        save_json(eval_metrics, 'eval_results_ensemble', args)

    if args.visualize:
        visualize(model, vis_dataloader, grad_cam_hooks, args)

    if args.plot_roc:
        # load results files from output_dir
        filenames = [f for f in os.listdir(args.output_dir) if f.startswith('eval_results') and f.endswith('.json')]
        if filenames == []: raise RuntimeError(
            'No `eval_results` files found in `{}` to plot results from.'.format(args.output_dir))
        # load and plot each
        for f in filenames:
            plot_roc(load_json(os.path.join(args.output_dir, f)), args, 'roc_pr_' + f.split('.')[0])

    writer.close()
