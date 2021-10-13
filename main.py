import os

import torch
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn

from dataloader.vehicle_data import DATASET
from utils.evaluate_utils import evaluate
from utils.im_utils import Compose, ToTensor, RandomHorizontalFlip
from utils.model_utils import create_model
from utils.train_utils import AverageMeter
from utils.im_utils import policy_v3
from utils.infinite_sample_utils import Infinite_sampler, reduce_dict
from Detector_RCNN_configuration import detectorConfig


class APP(object):
    """
    The training of the specified architecture will take infinite number of batches unless until it is specified. In order to evaluate the trained model at particular interval,
    the interval state is need to be mentioned.

    To start the training
    app = APP(args)
    app.run(args)

    The training history will be saved in save folder using tensorboard. To invoke the tensorfboarc '$ tensorboard --logdir='saved_dir'.
    The checkpoint will be saved periodically.
    """
    def __init__(self, auto_augment: bool, gpu_ids: str, model_save_dir: str, precision: str):

        """
        Args:
            :param auto_augment: To use AUTO-AUGMENT TECHNIQUES
            :param gpu_ids: Gpu ids, to be specified as string '0, 1, 2, 3'
            :param model_save_dir: Save directory of log file and checkpoint
            :param precision: Training precision

            TODO: Implementation of auto-augment technique
        """
        cudnn.benchmark = True
        if gpu_ids is not None:
            self.gpu_ids = [int(gpu_id) for gpu_id in gpu_ids.split(",")]
            print("Use GPU: {} for training".format(gpu_ids[:]))
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.gpu_ids = None
        self.precision = precision
        self.model_save_dir = model_save_dir
        self.device = torch.device("cuda", self.gpu_ids[0]) if self.gpu_ids is not None else torch.device("cuda")
        self.writer = SummaryWriter(os.path.join(model_save_dir, 'epoch_log'))
        Policy = policy_v3()
        if auto_augment:
            self.train_transform = Compose([ToTensor(),
                                            RandomHorizontalFlip(prob=0.4)],
                                            policies=Policy)
        else:
            self.train_transform = Compose([ToTensor(),
                                            RandomHorizontalFlip(prob=0.0)])
        self.test_transform = Compose([ToTensor()])

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    def get_dataLoader(self, dataroot: object, task: str, batch_size: int) -> object:
        """
        A commen data loader for training and testing.

        Args:
            :param dataroot (str): The data directory path. The structure of the directory should follow coco dataset pattern.
            :param task (str): train or test
            :param batch_size (int): Total batch size of data across all of the gpus.

        """
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        data_set = DATASET(dataroot, task, '2017', self.train_transform)
        sampler = Infinite_sampler(data_set.__len__())

        with torch.cuda.device(self.device):
            data_loader = torch.utils.data.DataLoader(data_set,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=nw,
                                                      sampler=sampler if task == 'train' else None,
                                                      collate_fn=self.collate_fn)

        return data_loader

    @staticmethod
    def get_lr(optimizer):
        """
        The function to return learning rate value whenever it is called.
        Args:
            :param optimizer: The optimizer of the current task.
            :return: Floating point of learning rate
        """
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def run(self, backbone: str, backbone_pretrained: str, max_iter: int, dataroot: str, num_class: int, per_batch_size: int,
            lr_config: object, resume_checkpoint: str, print_freq: int=50, save_ckpt_freq: int=1000) -> None:

        """
        Args:
            :param backbone (str): Architecture of the backbone module. Typically the name of the network. However, the network name must be given in the form of acceptable.
                                   Currently, supported backbones are mobilenet, resnet18, efficientnet-b0, transformer-CvT, nfnet-f0, resnet50_fpn, shufflenet.
            :param backbone_pretrained (str): Whether the backbone to use imagenet pretrained model or not
            :param RCNN_config (object): Configuration of the detector module.
            :param max_iter: The dataloader genenerate infinite number of batches of dataset with given batch size. To break the training loop, max_iter is used. If it is None,
                            the training loop will continue infinitely.
            :param dataroot (str): The data directory path. The structure of the directory should follow coco dataset pattern.
            :param num_class (int): Number of classes. Total number of classes should consider the background object. Ex. If the number of class is 10, num_class: int= 10 +1
            :param per_batch_size (int): Size of the batch size per GPU. Total number of batches across all GPU will be calculate automatically
            :param lr_config (object): All of the learning rate configuration. Including scheduler.
            :param resume_checkpoint (int): The path of trained model of backbone + detector. This is different from backbone_pretrained
            :param print_freq: Printing the training loss
            :param save_ckpt_freq: To save the checkpoint and evaluation step
            :return: None
        """

        global mAP
        ctx_num = len(self.gpu_ids) if self.gpu_ids is not None else 1
        batch_size = per_batch_size * ctx_num
        cudnn.benchmark = True
        trainLoader = self.get_dataLoader(dataroot=dataroot, task='train', batch_size=batch_size)
        testLodaer = self.get_dataLoader(dataroot=dataroot, task='test', batch_size=batch_size)

        model = create_model(backbone, detectorConfig(), num_class, backbone_pretrained)
        model.to(self.device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr_config.learningRate,
                                    momentum=lr_config.momentum, weight_decay=lr_config.Decay)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=lr_config.learningStep,
                                                       gamma=lr_config.learningGamma)
        scheduler_step = 1000
        if self.precision == "fp16":
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            print('Training with Mixed precision is confirmed')

        model = torch.nn.DataParallel(model, device_ids=self.gpu_ids, output_device=self.device).cuda(self.device)

        if resume_checkpoint != "":
            checkpoint = torch.load(resume_checkpoint)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            iteration = checkpoint['epoch'] + 1
            print("the training process from epoch{}...".format(iteration))
        else:
            iteration = 0

        train_loss = []
        val_mAPs = []
        best_mAP = 0

        loss_classifier = AverageMeter()
        loss_box_reg = AverageMeter()
        loss_objectness = AverageMeter()
        loss_rpn_box_reg = AverageMeter()

        while True:

            model.train()
            images, targets = next(iter(trainLoader))
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()
            if isinstance(train_loss, list):
                train_loss.append(loss_value)
            optimizer.zero_grad()

            if precision == "fp16":
                with amp.scale_loss(losses, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                losses.backward()
            optimizer.step()

            if iteration % scheduler_step == 0 and iteration != 0:
                lr_scheduler.step()

            loss_classifier.update(loss_dict_reduced['loss_classifier'].item())
            loss_box_reg.update(loss_dict_reduced['loss_box_reg'].item())
            loss_objectness.update(loss_dict_reduced['loss_objectness'].item())
            loss_rpn_box_reg.update(loss_dict_reduced['loss_rpn_box_reg'].item())

            if iteration % print_freq == 0:
                print('Train : [Iter %d] \t Class Loss %.3f, Box Reg Loss %.3f, Objectness Loss %.3f, RPN Box Reg Loss %.3f, LR %.5f' %
                      (iteration, loss_classifier.avg, loss_box_reg.avg, loss_objectness.avg, loss_rpn_box_reg.avg, self.get_lr(optimizer)))

            if iteration % save_ckpt_freq == 0:
                _, mAP = evaluate(model, testLodaer, device=self.device, mAP_list=val_mAPs)
                print('validation mAp is {}'.format(mAP))
                print('best mAp is {}'.format(best_mAP))

                checkpoint_info = {'lr': optimizer.param_groups[0]['lr'],
                                   'val_mAP': mAP}
                for k, v in loss_dict.items():
                    checkpoint_info[k] = v.item()

                if mAP > best_mAP:
                    best_mAP = mAP
                    save_files = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': iteration}
                    self.model_save_dir = self.model_save_dir
                    os.makedirs(model_save_dir, exist_ok=True)
                    torch.save(save_files,
                               os.path.join(model_save_dir, "{}-model-{}-mAp-{}.pth".format(backbone, iteration, mAP)))

            iteration += 1
            self.writer.close()

            if iteration == max_iter:
                break


if __name__ == '__main__':
    class learningRateConfig:
        learningRate = 5e-3
        momentum = 0.9
        Decay = 0.0005
        learningStep = 100
        learningGamma = 0.33

    backbone = 'resnet18'  # [mobilenet, resnet18, efficientnet-b0, transformer-CvT, nfnet-f0, resnet50_fpn, shufflenet]
    backbone_pretrained = ''
    RCNN_config = detectorConfig()
    dataroot = '/media/raja/6TB/Own_Trials/Dataset/Kaggle_Modified/Named_data'

    max_iter = 10

    num_class = 21 + 1
    per_batch_size = 8
    resume_checkpoint = ''

    auto_augment = False
    gpu_ids = '0'
    model_save_dir = '{}_checkpoint'.format(backbone)
    precision = 'fp32'

    app = APP(auto_augment, gpu_ids, model_save_dir, precision)
    app.run(backbone, backbone_pretrained, max_iter, dataroot, num_class, per_batch_size, learningRateConfig(),
            resume_checkpoint)
