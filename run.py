import logging
import os
import random
from argparse import Namespace
from inspect import Parameter
from typing import Any, Iterator

import numpy as np
import torch
from torch import distributed, optim
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler

import argparser
import tasks
import utils
from apex import amp
from apex.parallel import DistributedDataParallel
from dataset import (AdeSegmentationIncremental, PotsdamIncremental, VaihingenIncremental, VOCSegmentationIncremental,
                     transform)
from metrics import StreamSegMetrics
from segmentation_module import make_model
from train import Trainer
from utils.logger import Logger


def save_ckpt(path, model, trainer, optimizer, scheduler, epoch, best_score):
    """ save current model
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
        "trainer_state": trainer.state_dict()
    }
    torch.save(state, path)


def get_augmentations(opts: Namespace):
    if opts.dataset in ("vaihingen", "potsdam"):
        # internal not to import for nothing
        import albumentations as alb
        from albumentations.pytorch.transforms import ToTensorV2

        if not opts.pretrained:
            means = [0.332, 0.361, 0.335, 0.383]
            stds = [0.143, 0.140, 0.145, 0.143]
        else:
            means = [0.485, 0.456, 0.406, 0.485]
            stds = [0.229, 0.224, 0.225, 0.229]
        # slice the vectors depending on whether we require IR or not
        means = means[:opts.input_channels]
        stds = stds[:opts.input_channels]
        if opts.include_dsm:
            means += [0.179]
            stds += [0.216]

        img_size = opts.crop_size
        half_size = opts.crop_size // 2
        train_transform = alb.Compose([
            alb.RandomSizedCrop(min_max_height=(half_size, img_size), height=img_size, width=img_size, p=0.5),
            alb.Flip(p=0.5),
            alb.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.5, rotate_limit=90, p=0.5),
            alb.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.8),
            alb.Normalize(mean=means, std=stds),
            ToTensorV2()
        ])
        val_transform = alb.Compose([alb.Normalize(mean=means, std=stds), ToTensorV2()])
    else:
        train_transform = transform.Compose([
            transform.RandomResizedCrop(opts.crop_size, (0.5, 2.0)),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if opts.crop_val:
            val_transform = transform.Compose([
                transform.Resize(size=opts.crop_size),
                transform.CenterCrop(size=opts.crop_size),
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # no crop, batch size = 1
            val_transform = transform.Compose([
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    return train_transform, val_transform


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform, val_transform = get_augmentations(opts)

    labels, labels_old, path_base = tasks.get_task_labels(opts.dataset, opts.task, opts.step)
    labels_cum = labels_old + labels

    optional_args = dict()

    if opts.dataset == 'voc':
        dataset = VOCSegmentationIncremental
    elif opts.dataset == 'ade':
        dataset = AdeSegmentationIncremental
    # ! insert my dataset
    elif opts.dataset == "potsdam":
        dataset = PotsdamIncremental
        optional_args["channels"] = opts.input_channels
        optional_args["include_dsm"] = opts.include_dsm
    elif opts.dataset == "vaihingen":
        dataset = VaihingenIncremental
        optional_args["channels"] = opts.input_channels
        optional_args["include_dsm"] = opts.include_dsm
    else:
        raise NotImplementedError

    if opts.overlap:
        path_base += "-ov"

    if not os.path.exists(path_base):
        os.makedirs(path_base, exist_ok=True)

    train_dst = dataset(root=opts.data_root,
                        train=True,
                        transform=train_transform,
                        labels=list(labels),
                        labels_old=list(labels_old),
                        idxs_path=path_base + f"/train-{opts.step}.npy",
                        masking=not opts.no_mask,
                        overlap=opts.overlap,
                        **optional_args)

    if not opts.no_cross_val:    # if opts.cross_val:
        train_len = int(0.8 * len(train_dst))
        val_len = len(train_dst) - train_len
        train_dst, val_dst = torch.utils.data.random_split(train_dst, [train_len, val_len])
    else:    # don't use cross_val
        val_dst = dataset(root=opts.data_root,
                          train=False,
                          transform=val_transform,
                          labels=list(labels),
                          labels_old=list(labels_old),
                          idxs_path=path_base + f"/val-{opts.step}.npy",
                          masking=not opts.no_mask,
                          overlap=True,
                          **optional_args)

    test_dst = dataset(root=opts.data_root,
                       train=opts.val_on_trainset,
                       transform=val_transform,
                       labels=list(labels_cum),
                       labels_old=None,
                       idxs_path=path_base + f"/test-{opts.step}.npy",
                       masking=True,
                       overlap=True,
                       **optional_args)

    # ? no test set used, only validation?
    return train_dst, val_dst, test_dst, len(labels_cum)


def get_optimizer(opts: Namespace, parameters: Iterator[Parameter]) -> optim.Optimizer:
    if opts.optim == "sgd":
        return torch.optim.SGD(parameters, lr=opts.lr, momentum=0.9, nesterov=True)
    elif opts.optim == "adam":
        return optim.Adam(parameters, lr=opts.lr)
    elif opts.optim == "adamw":
        return optim.AdamW(parameters, lr=opts.lr)
    else:
        raise NotImplementedError(f"Optimizer '{opts.optim}' not implemented")


def get_scheduler(opts: Namespace, train_loader: data.DataLoader, optimizer: optim.Optimizer) -> Any:
    if opts.lr_policy == 'poly':
        return utils.PolyLR(optimizer, max_iters=opts.epochs * len(train_loader), power=opts.lr_power)
    elif opts.lr_policy == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    elif opts.lr_policy == 'cos':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    else:
        raise NotImplementedError(f"Scheduler '{opts.lr_policy}' not implemented")


def load_old_model(opts: Namespace,
                   new_model: torch.nn.Module,
                   old_model: torch.nn.Module,
                   task_name: str,
                   logger: Any,
                   device: torch.device = "cpu"):
    if opts.step_ckpt is not None:
        path = opts.step_ckpt
    else:
        path = f"checkpoints/step/{task_name}_{opts.name}_{opts.step - 1}.pth"

    # generate model from path
    if os.path.exists(path):
        step_checkpoint = torch.load(path, map_location="cpu")
        new_model.load_state_dict(step_checkpoint['model_state'], strict=False)    # False because of incr. classifiers
        if opts.init_balanced:
            # implement the balanced initialization (new cls has weight of background and bias = bias_bkg - log(N+1)
            new_model.module.init_new_classifier(device)
        # Load state dict from the model state dict, that contains the old model parameters
        old_model.load_state_dict(step_checkpoint['model_state'], strict=True)    # Load also here old parameters
        logger.info(f"[!] Previous model loaded from {path}")
        # clean memory
        del step_checkpoint['model_state']
    elif opts.debug:
        logger.info(f"[!] WARNING: Unable to find of step {opts.step - 1}! Do you really want to do from scratch?")
    else:
        raise FileNotFoundError(path)
    # put the old model into distributed memory and freeze it
    for par in old_model.parameters():
        par.requires_grad = False
    return new_model, old_model


def load_state(trainer: Any, model: torch.nn.Module, optimizer: optim.Optimizer, scheduler: Any, logger: Any,
               checkpoint_path: str) -> torch.nn.Module:
    checkpoint = torch.load(opts.ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    cur_epoch = checkpoint["epoch"] + 1
    best_score = checkpoint['best_score']
    logger.info("[!] Model restored from %s" % opts.ckpt)
    # if we want to resume training, resume trainer from checkpoint
    if 'trainer_state' in checkpoint:
        trainer.load_state_dict(checkpoint['trainer_state'])
    del checkpoint
    return cur_epoch, best_score


def main(opts):
    logging.basicConfig(level=logging.INFO)
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = opts.local_rank, torch.device(opts.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    # Initialize logging
    task_name = f"{opts.task}-{opts.dataset}"
    logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/"
    if rank == 0:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step)
    else:
        logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=False)
    logger.print(f"Device: {device}")

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # xxx Set up dataloader
    train_dst, val_dst, test_dst, n_classes = get_dataset(opts)
    # reset the seed, this revert changes in random seed
    random.seed(opts.random_seed)

    train_loader = data.DataLoader(train_dst,
                                   batch_size=opts.batch_size,
                                   sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
                                   num_workers=opts.num_workers,
                                   drop_last=True)
    val_loader = data.DataLoader(val_dst,
                                 batch_size=opts.batch_size if opts.crop_val else 1,
                                 sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
                                 num_workers=opts.num_workers)
    logger.info(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)},"
                f" Test set: {len(test_dst)}, n_classes {n_classes}")
    logger.info(f"Total batch size is {opts.batch_size * world_size}")

    # xxx Set up model
    logger.info(f"Backbone: {opts.backbone}")

    step_checkpoint = None
    model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
    logger.info(f"[!] Model made with{'' if opts.pretrained else 'out'} pre-trained")

    if opts.step == 0:    # if step 0, we don't need to instance the model_old
        model_old = None
    else:    # instance model_old
        model_old = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1))

    if opts.fix_bn:
        model.fix_bn()
    logger.debug(model)

    # xxx Set up optimizer
    params = []
    if not opts.freeze:
        params.append({
            "params": filter(lambda p: p.requires_grad, model.body.parameters()),
            'weight_decay': opts.weight_decay
        })
    params.append({
        "params": filter(lambda p: p.requires_grad, model.head.parameters()),
        'weight_decay': opts.weight_decay
    })
    params.append({
        "params": filter(lambda p: p.requires_grad, model.classifiers.parameters()),
        'weight_decay': opts.weight_decay
    })

    optimizer = get_optimizer(opts, parameters=params)
    scheduler = get_scheduler(opts, train_loader, optimizer)

    if model_old is not None:
        (model, model_old), optimizer = amp.initialize((model.to(device), model_old.to(device)),
                                                       optimizer=optimizer,
                                                       opt_level=opts.opt_level)
        model_old = DistributedDataParallel(model_old)
    else:
        model, optimizer = amp.initialize(model.to(device), optimizer, opt_level=opts.opt_level)
    model = DistributedDataParallel(model, delay_allreduce=True)

    # Load old model from old weights if step > 0!
    if opts.step > 0:
        model, model_old = load_old_model(opts,
                                          new_model=model,
                                          old_model=model_old,
                                          task_name=task_name,
                                          logger=logger,
                                          device=device)
        model_old.eval()

    # xxx Set up Trainer
    trainer_state = None
    # if not first step, then instance trainer from step_checkpoint
    if opts.step > 0 and step_checkpoint is not None:
        trainer_state = step_checkpoint.get('trainer_state')

    # instance trainer (model must have already the previous step weights)
    trainer = Trainer(model,
                      model_old,
                      device=device,
                      opts=opts,
                      trainer_state=trainer_state,
                      classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))

    # Handle checkpoint for current model (model old will always be as previous step or None)
    best_score = 0.0
    cur_epoch = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        cur_epoch, best_score = load_state(trainer, model, optimizer, scheduler, logger, opts.ckpt)
    else:
        if opts.step == 0:
            logger.info("[!] Train from scratch")

    # xxx Train procedure
    # print opts before starting training to log all parameters
    logger.add_table("Opts", vars(opts))

    sample_ids = None
    if rank == 0 and opts.sample_num > 0:
        # sample indices for visualization purposes during training
        sample_ids = np.random.choice(len(val_loader), opts.sample_num, replace=False)
        logger.info(f"The samples id are {sample_ids}")
    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))    # convert labels to images
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    TRAIN = not opts.test
    val_metrics = StreamSegMetrics(n_classes)
    results = {}

    # check if random is equal here.
    logger.print(torch.randint(0, 100, (1, 1)))
    # train/val here
    while cur_epoch < opts.epochs and TRAIN:
        # =====  Train  =====
        model.train()
        epoch_loss = trainer.train(cur_epoch=cur_epoch,
                                   optim=optimizer,
                                   train_loader=train_loader,
                                   scheduler=scheduler,
                                   logger=logger)

        logger.info(f"End of Epoch {cur_epoch}/{opts.epochs}, Average Loss={epoch_loss[0]+epoch_loss[1]},"
                    f" Class Loss={epoch_loss[0]}, Reg Loss={epoch_loss[1]}")

        # =====  Log metrics on Tensorboard =====
        logger.add_scalar("E-Loss", epoch_loss[0] + epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-reg", epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-cls", epoch_loss[0], cur_epoch)

        # =====  Validation  =====
        if (cur_epoch + 1) % opts.val_interval == 0:
            logger.info("validate on val set...")
            model.eval()
            val_loss, val_score, ret_samples = trainer.validate(loader=val_loader,
                                                                metrics=val_metrics,
                                                                ret_samples_ids=sample_ids,
                                                                logger=logger)

            logger.print("Done validation")
            logger.info(f"End of Validation {cur_epoch}/{opts.epochs}, Validation Loss={val_loss[0]+val_loss[1]},"
                        f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}")

            logger.info(val_metrics.to_str(val_score))

            # =====  Save Best Model  =====
            if rank == 0:    # save best model at the last iteration
                score = val_score['Mean IoU']
                # best model to build incremental steps
                save_ckpt(f"checkpoints/step/{task_name}_{opts.name}_{opts.step}.pth", model, trainer, optimizer,
                          scheduler, cur_epoch, score)
                logger.info("[!] Checkpoint saved.")

            # =====  Log metrics on Tensorboard =====
            # visualize validation score and samples
            logger.add_scalar("V-Loss", val_loss[0] + val_loss[1], cur_epoch)
            logger.add_scalar("V-Loss-reg", val_loss[1], cur_epoch)
            logger.add_scalar("V-Loss-cls", val_loss[0], cur_epoch)
            logger.add_scalar("Val_Overall_Acc", val_score['Overall Acc'], cur_epoch)
            logger.add_scalar("Val_MeanIoU", val_score['Mean IoU'], cur_epoch)
            logger.add_table("Val_Class_IoU", val_score['Class IoU'], cur_epoch)
            logger.add_scalar("Val_MeanF1", val_score["Mean F1"], cur_epoch)
            logger.add_table("Val_Class_F1", val_score['Class F1'], cur_epoch)
            logger.add_table("Val_Acc_IoU", val_score['Class Acc'], cur_epoch)
            # logger.add_figure("Val_Confusion_Matrix", val_score['Confusion Matrix'], cur_epoch)

            # keep the metric to print them at the end of training
            results["V-IoU"] = val_score['Class IoU']
            results["V-Acc"] = val_score['Class Acc']
            results["V-F1"] = val_score["Class F1"]

            for k, (img, target, lbl) in enumerate(ret_samples):
                img = (denorm(img[:3]) * 255).astype(np.uint8)
                target = label2color(target).transpose(2, 0, 1).astype(np.uint8)
                lbl = label2color(lbl).transpose(2, 0, 1).astype(np.uint8)

                concat_img = np.concatenate((img, target, lbl), axis=2)    # concat along width
                logger.add_image(f'Sample_{k}', concat_img, cur_epoch)

        cur_epoch += 1

    # =====  Save Best Model at the end of training =====
    if rank == 0 and TRAIN:    # save best model at the last iteration
        # best model to build incremental steps
        save_ckpt(f"checkpoints/step/{task_name}_{opts.name}_{opts.step}.pth", model, trainer, optimizer, scheduler,
                  cur_epoch, best_score)
        logger.info("[!] Checkpoint saved.")

    torch.distributed.barrier()

    # xxx From here starts the test code
    logger.info("*** Test the model on all seen classes...")
    # make data loader
    test_loader = data.DataLoader(test_dst,
                                  batch_size=opts.batch_size if opts.crop_val else 1,
                                  sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
                                  num_workers=opts.num_workers)

    # load best model
    if TRAIN:
        model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
        # Put the model on GPU
        model = DistributedDataParallel(model.cuda(device))
        ckpt = f"checkpoints/step/{task_name}_{opts.name}_{opts.step}.pth"
        checkpoint = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        logger.info(f"*** Model restored from {ckpt}")
        del checkpoint
        trainer = Trainer(model, None, device=device, opts=opts)

    model.eval()

    val_loss, val_score, _ = trainer.validate(loader=test_loader, metrics=val_metrics, logger=logger)
    logger.print("Done test")
    logger.info(f"*** End of Test, Total Loss={val_loss[0]+val_loss[1]},"
                f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}")
    logger.info(val_metrics.to_str(val_score))
    logger.add_table("Test_Class_IoU", val_score['Class IoU'])
    logger.add_table("Test_Class_Acc", val_score['Class Acc'])
    logger.add_table("Test_Class_F1", val_score["Class F1"])
    logger.add_figure("Test_Confusion_Matrix", val_score['Confusion Matrix'])
    results["T-IoU"] = val_score['Class IoU']
    results["T-Acc"] = val_score['Class Acc']
    results["T-F1"] = val_score["Class F1"]
    logger.add_results(results)

    logger.add_scalar("T_Overall_Acc", val_score['Overall Acc'], opts.step)
    logger.add_scalar("T_MeanIoU", val_score['Mean IoU'], opts.step)
    logger.add_scalar("T_MeanAcc", val_score['Mean Acc'], opts.step)
    logger.add_scalar("T_MeanF1", val_score["Mean F1"], opts.step)

    logger.close()


if __name__ == '__main__':
    parser = argparser.get_argparser()
    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)
    os.makedirs("checkpoints/step", exist_ok=True)
    main(opts)
