from .coresetmethod import CoresetMethod
import torch, time, os
from torch import nn
import numpy as np
from copy import deepcopy
from .. import nets
from torchvision import transforms
from transformers import (
    AutoProcessor, AutoImageProcessor, Dinov2Model, SiglipVisionModel, SiglipImageProcessor,
    AutoModel, CLIPImageProcessor, ViTImageProcessor, ViTModel
)

class EarlyTrain(CoresetMethod):
    '''
    Core code for training related to coreset selection methods when pre-training is required.
    '''

    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None,
                 torchvision_pretrain: bool = False, dst_pretrain_dict: dict = {}, fraction_pretrain=1., dst_test=None,
                 **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.specific_model = specific_model
        self.im_size_pretrain = args.im_size

        if fraction_pretrain <= 0. or fraction_pretrain > 1.:
            raise ValueError("Illegal pretrain fraction value.")
        self.fraction_pretrain = fraction_pretrain

        if dst_pretrain_dict.__len__() != 0:
            dict_keys = dst_pretrain_dict.keys()
            if 'im_size' not in dict_keys or 'channel' not in dict_keys or 'dst_train' not in dict_keys or \
                    'num_classes' not in dict_keys or 'dst_test' not in dict_keys:
                raise AttributeError(
                    'Argument dst_pretrain_dict must contain imszie, channel, dst_train, dst_test and num_classes.')
            if dst_pretrain_dict['channel'] != args.channel:
                raise ValueError("channel of pretrain dataset does not match that of the training dataset.")
            # if dst_pretrain_dict['num_classes'] != args.num_classes:
            #     self.num_classes_mismatch()
            if dst_pretrain_dict['im_size'][0] != args.im_size[0]:
                self.im_size_pretrain = dst_pretrain_dict['im_size'] #used when pretrained datasets are not the target full set
            #     raise ValueError("im_size of pretrain dataset does not match that of the training dataset.")

        self.dst_pretrain_dict = dst_pretrain_dict
        self.torchvision_pretrain = torchvision_pretrain
        self.if_dst_pretrain = (len(self.dst_pretrain_dict) != 0)

        if torchvision_pretrain:
            # Pretrained models in torchvision only accept 224*224 inputs, therefore we resize current
            # datasets to 224*224.
            if args.im_size[0] != 224 or args.im_size[1] != 224:
                self.dst_train = deepcopy(dst_train)
                self.dst_train.transform = transforms.Compose([self.dst_train.transform, transforms.Resize(224)])
                if self.if_dst_pretrain:
                    self.dst_pretrain_dict['dst_train'] = deepcopy(dst_pretrain_dict['dst_train'])
                    self.dst_pretrain_dict['dst_train'].transform = transforms.Compose(
                        [self.dst_pretrain_dict['dst_train'].transform, transforms.Resize(224)])
        
        if self.if_dst_pretrain:
            self.n_pretrain = len(self.dst_pretrain_dict['dst_train'])
        self.n_pretrain_size = round(
            self.fraction_pretrain * (self.n_pretrain if self.if_dst_pretrain else self.n_train))
        self.dst_test = self.dst_pretrain_dict['dst_test'] if self.if_dst_pretrain else dst_test

    def train(self, epoch, list_of_train_idx, **kwargs):
        """ Train model for one epoch """

        self.before_train()
        self.model.train()

        print('\n=> Training Epoch #%d' % epoch)
        trainset_permutation_inds = np.random.permutation(list_of_train_idx)
        batch_sampler = torch.utils.data.BatchSampler(trainset_permutation_inds, batch_size=self.args.selection_batch,
                                                      drop_last=False)
        trainset_permutation_inds = list(batch_sampler)

        train_loader = torch.utils.data.DataLoader(self.dst_pretrain_dict['dst_train'] if self.if_dst_pretrain
                                                   else self.dst_train, shuffle=False, batch_sampler=batch_sampler,
                                                   num_workers=self.args.workers, pin_memory=True)

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            # Forward propagation, compute loss, get predictions
            self.model_optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.after_loss(outputs, loss, targets, trainset_permutation_inds[i], epoch)

            # Update loss, backward propagate, update optimizer
            loss = loss.mean()

            self.while_update(outputs, loss, targets, epoch, i, self.args.selection_batch)

            loss.backward()
            self.model_optimizer.step()
        return self.finish_train()

    def run(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.train_indx = np.arange(self.n_train) #used in selection method

        # Setup model and loss
        self.model_name = self.args.model if self.specific_model is None else self.specific_model
        print('self.model_name=',self.model_name)

        # Load pre-trained or custom model
        if self.model_name == 'CLIP':
            print('using CLIP API (ViT-L/14):')
            self.model, self.preprocess = nets.__dict__['CLIP']('ViT-L/14', self.args.device)
            self.dst_train.transform = self.preprocess
        elif self.model_name == 'DINOV2':
            print('using DINOv2-B (dinov2_vits14) pre-trained vision model:')
            #---online load
            # self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.args.device)

            #---offline load
            self.model = Dinov2Model.from_pretrained('./pretrain/dinov2-main').to(self.args.device)
            if self.args.im_size[0]%14 != 0:
                print('self.args.im_size[0]%14 != 0')
                self.dst_train.transform = transforms.Compose([transforms.RandomResizedCrop(224, antialias=True), 
                                                                self.dst_train.transform]) 
        elif self.model_name == 'SIGLIP':
            print('using siglip-base-patch16-224 pre-trained vision model:')
            #---online load
            # self.model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(self.args.device)
            # self.dst_train.transform = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224", image_mean=self.args.mean, image_std=self.args.std)

            #---offline load
            self.model = SiglipVisionModel.from_pretrained("./pretrain/siglip-base-patch16-224").to(self.args.device)
            self.dst_train.transform = SiglipImageProcessor.from_pretrained("./pretrain/siglip-base-patch16-224", image_mean=self.args.mean, image_std=self.args.std)
        elif self.model_name == 'EVA-CLIP':
            print('using BAAI/EVA-CLIP-8B pre-trained vision model:')
            #---online load
            # self.model = AutoModel.from_pretrained('BAAI/EVA-CLIP-8B', 
            #                                         torch_dtype=torch.float16,
            #                                         trust_remote_code=True).to(self.args.device)
            # self.dst_train.transform = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", image_mean=self.args.mean, image_std=self.args.std)

            #---offline load
            self.model = AutoModel.from_pretrained('./pretrain/EVA-CLIP-8B', 
                                                    torch_dtype=torch.float16,
                                                    trust_remote_code=True).to(self.args.device)
            self.dst_train.transform = CLIPImageProcessor.from_pretrained("./pretrain/clip-vit-large-patch14", image_mean=self.args.mean, image_std=self.args.std)
        else:
            if self.dst_pretrain_dict.__len__() != 0:
                self.model = nets.__dict__[self.model_name](
                    self.args.channel, self.dst_pretrain_dict["num_classes"] if self.if_dst_pretrain else self.num_classes,
                    pretrained=self.torchvision_pretrain,
                    im_size=(224, 224)).to(self.args.device)
            else:
                self.model = nets.__dict__[self.model_name](
                    self.args.channel, self.dst_pretrain_dict["num_classes"] if self.if_dst_pretrain else self.num_classes,
                    pretrained=self.torchvision_pretrain,
                    im_size=(224, 224) if self.torchvision_pretrain else self.args.im_size).to(self.args.device)

        if self.args.device == "cpu":
            print("Using CPU.")
        else:
            if self.args.gpu is not None:
                torch.cuda.set_device(self.args.gpu[0])
                device_ids = self.args.gpu if isinstance(self.args.gpu, (list, tuple)) else [self.args.gpu]
                self.model = nets.nets_utils.MyDataParallel(self.model, device_ids=device_ids).to(self.args.device)
            elif torch.cuda.device_count() > 1:
                self.model = nets.nets_utils.MyDataParallel(self.model).cuda()
            else:
                self.model = self.model.to(self.args.device)

        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.criterion.__init__()

        # Setup optimizer
        if self.model_name not in ['CLIP', 'DINOV2', 'SIGLIP', 'EVA-CLIP']:
            print('Setup optimizer')
            if self.args.selection_optimizer == "SGD":
                self.model_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.selection_lr,
                                                    momentum=self.args.selection_momentum,
                                                    weight_decay=self.args.selection_weight_decay,
                                                    nesterov=self.args.selection_nesterov)
            elif self.args.selection_optimizer == "Adam":
                self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.selection_lr,
                                                        weight_decay=self.args.selection_weight_decay)
            else:
                self.model_optimizer = torch.optim.__dict__[self.args.selection_optimizer](self.model.parameters(),
                                                                        lr=self.args.selection_lr,
                                                                        momentum=self.args.selection_momentum,
                                                                        weight_decay=self.args.selection_weight_decay,
                                                                        nesterov=self.args.selection_nesterov)
        self.before_run()
        self.before_epoch()

        for epoch in range(self.epochs):
            list_of_train_idx = np.random.choice(np.arange(self.n_pretrain if self.if_dst_pretrain else self.n_train),
                                                 self.n_pretrain_size, replace=False)
            self.train(epoch, list_of_train_idx)

            if self.dst_test is not None and self.args.selection_test_interval > 0 and (
                    epoch + 1) % self.args.selection_test_interval == 0:
                self.test(epoch)

            if self.args.save_path != "" and (self.epochs > 0) and (epoch+1==self.epochs) and (self.specific_model in ['ResNet18', 'ResNet50']):
                torch.save(self.model.state_dict(),
                            os.path.join(self.args.save_path, "pretrained_{}_{}epoch_{}.pth".format(self.specific_model, self.epochs, self.text_acc)))
                print("=> [Pre-trained model] Saving checkpoint for epoch %d, with Prec@1 %f." % (self.epochs, self.text_acc))
        
        self.after_epoch()
        if self.epochs == 0 and (self.specific_model in ['ResNet18', 'ResNet50']):
            self.test()

        return self.finish_run()

    def test(self, epoch=0):
        self.model.no_grad = True
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(self.dst_test if self.args.selection_test_fraction == 1. else
                                                  torch.utils.data.Subset(self.dst_test, np.random.choice(
                                                      np.arange(len(self.dst_test)),
                                                      round(len(self.dst_test) * self.args.selection_test_fraction),
                                                      replace=False)),
                                                  batch_size=self.args.selection_batch, shuffle=False,
                                                  num_workers=self.args.workers, pin_memory=True)
        correct = 0.
        total = 0.

        print('\n=> Testing Epoch #%d' % epoch)

        for batch_idx, (input, target) in enumerate(test_loader):
            inputs, targets = input.to(self.args.device), target.to(self.args.device)
            output = self.model(inputs)
            loss = self.criterion(output, targets).sum()

            predicted = torch.max(output.data, 1).indices.cpu()
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % self.args.print_freq == 0:
                print('| Test Epoch [%3d/%3d] Iter[%3d/%3d]\t\tTest Loss: %.4f Test Acc: %.3f%%' % (
                    epoch, self.epochs, batch_idx + 1, (round(len(self.dst_test) * self.args.selection_test_fraction) //
                                                        self.args.selection_batch) + 1, loss.item(),
                    100. * correct / total))

        print('| Test Epoch [%3d/%3d] \t\tTest Loss: %.4f Test Acc: %.3f%%' % (
            epoch, self.epochs, loss.item(), 100. * correct / total))

        self.text_acc = 100. * correct / total
        print(f"| Test Epoch [{epoch}/{self.epochs}] Final Test Acc: {self.text_acc:.2f}%")
        self.model.no_grad = False

    def num_classes_mismatch(self):
        pass

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        pass

    def finish_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def finish_run(self):
        pass

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result