from .earlytrain import EarlyTrain
import torch
import numpy as np
from .methods_utils import euclidean_dist
from ..nets.nets_utils import MyDataParallel
from datetime import datetime
import os
import collections
from torch import tensor
from .. import nets
from torchvision import transforms
from transformers import (
    AutoModel, CLIPImageProcessor, Dinov2Model,
    SiglipVisionModel, SiglipImageProcessor
)

def get_distance_rank(feats_list, num_lvms):
    """
    feats_list: List[np.ndarray], each of shape [N, D_i] where D_i may vary
    Returns:
        ranks: np.ndarray of shape [num_lvms, N], where ranks[i][j] = rank of sample j in model i
        
        Note: ranks are from 0 (closest to center) to N-1 (farthest from center)
        np.argsort(distances) returns indices that sort the array
        np.argsort(np.argsort(distances)) returns the ranking (i.e., position in sorted order)
    """
    ranks_list = []

    for feats in feats_list:  # feats: [N, D_i]
        prot = np.mean(feats, axis=0, keepdims=True)  # [1, D_i]
        dists = np.linalg.norm(feats - prot, axis=1)  # [N]
        ranks = np.argsort(np.argsort(dists))  # double argsort: ranks from 0 (closest) to N-1 (farthest)
        ranks_list.append(ranks)

    return np.stack(ranks_list, axis=0)  # [num_lvms, N]

def weight_functions(sampling_rate, a, k=1, sr=0.5):
    W1 = a + (1 - a) * (1 / (1 + np.exp(k * (sampling_rate - sr))))
    W2 = 1 - W1
    return W1, W2

def get_selected_idx(sampling_rate, ranks_np, APL):
    num_data = ranks_np.shape[1]
    budget = round(num_data*sampling_rate)
    rank_mean = np.mean(ranks_np, axis=0)
    RAM = rank_mean/num_data

    a = 0.2 #modify
    k = 1 #modify
    km, kc = weight_functions(sampling_rate, a, k)
    candidate_score = km*RAM + kc*APL
    
    sorted_idx = np.argsort(candidate_score) # small - large
    ids = sorted_idx[:budget]
    return ids

class RAM_APL(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=0,
                 specific_model="ResNet18", balance: bool = False, already_selected=[],
                 torchvision_pretrain: bool = False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs=epochs, specific_model=specific_model,
                         torchvision_pretrain=torchvision_pretrain, **kwargs)

        if already_selected.__len__() != 0:
            if min(already_selected) < 0 or max(already_selected) >= self.n_train:
                raise ValueError("List of already selected points out of the boundary.")
        self.already_selected = np.array(already_selected)
        self.balance = balance
        self.transform = self.dst_train.transform

        self.pretrained_model_paths = { # offline local path
            'DINOV2': './pretrain/dinov2-main',
            'SIGLIP': './pretrain/siglip-base-patch16-224',
            'EVA-CLIP': './pretrain/EVA-CLIP-8B'
        }

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))
    
    def init_LMs(self, model_names=[]):
        for name in model_names:
            print(f'Initializing backbone: {name}')
            if name == 'CLIP':
                self.model_CLIP, self.preprocess_CLIP = nets.__dict__['CLIP']('ViT-L/14', self.args.device)
            elif name == 'DINOV2':
                self.model_DINOV2 = Dinov2Model.from_pretrained(self.pretrained_model_paths['DINOV2']).to(self.args.device)
            elif name == 'SIGLIP':
                self.model_SIGLIP = SiglipVisionModel.from_pretrained(self.pretrained_model_paths['SIGLIP'], torch_dtype=torch.float16).to(self.args.device)
            elif name == 'EVA-CLIP':
                self.model_EVACLIP = AutoModel.from_pretrained(self.pretrained_model_paths['EVA-CLIP'], torch_dtype=torch.float16, trust_remote_code=True).to(self.args.device)
            else:
                raise ValueError(f"Unsupported model name: {name}")

    def construct_matrix(self, model_name=None, index=None):
        if model_name == 'CLIP':
            self.dst_train.transform = self.preprocess_CLIP
        elif model_name == 'DINOV2':
            if self.args.im_size[0]%14 != 0:
                print('int(self.args.im_size[0]/14) != 0')
                self.dst_train.transform = transforms.Compose([transforms.RandomResizedCrop(224, antialias=True), 
                                                                self.dst_train.transform])
        elif model_name == 'SIGLIP':
            self.dst_train.transform = SiglipImageProcessor.from_pretrained(self.pretrained_model_paths['SIGLIP'], image_mean=self.args.mean, image_std=self.args.std)
        elif model_name == 'EVA-CLIP':
            self.dst_train.transform = CLIPImageProcessor.from_pretrained("./pretrain/clip-vit-large-patch14", image_mean=self.args.mean, image_std=self.args.std)

        data_loader = torch.utils.data.DataLoader(self.dst_train if index is None else
                                        torch.utils.data.Subset(self.dst_train, index),
                                        shuffle=False,
                                        batch_size=self.args.selection_batch,
                                        num_workers=self.args.workers)
          
        model_attr = f'model_{model_name.replace("-", "").upper()}'
        model = getattr(self, model_attr)
        model.eval()

        features, labels = [], []
        with torch.no_grad():
            for inputs, targets in data_loader:
                if model_name == 'CLIP':
                    features_batch = model.encode_image(inputs.to(self.args.device))
                elif model_name == 'DINOV2':
                    outputs = model(inputs.to(self.args.device))
                    features_batch = outputs.pooler_output
                elif model_name == 'SIGLIP':
                    inputs = inputs['pixel_values'][0].to(torch.float16)
                    outputs = model(inputs.to(self.args.device))
                    features_batch = outputs.pooler_output
                elif model_name == 'EVA-CLIP':
                    inputs = inputs['pixel_values'][0].to(torch.float16)
                    features_batch = model.encode_image(inputs.to(self.args.device))
                else:
                    outputs = model(inputs.to(self.args.device))
                    features_batch = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs

                features.append(features_batch.cpu().float().numpy())
                labels.append(targets.numpy())

        return np.vstack(features), np.concatenate(labels)

    def before_run(self):
        pass

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module          
            
    def finish_selection(self, model_names):
        for name in model_names:
            attr_name = f'model_{name.replace("-", "").upper()}'
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        torch.cuda.empty_cache()

    def select(self, **kwargs):
        self.run()

        model_names = ['CLIP', 'DINOV2']
        self.init_LMs(model_names)
        
        if self.balance:
            model_features = {}
            for model_name in model_names:
                feats, labels = self.construct_matrix(model_name)
                model_features[model_name] = {'feats': feats, 'labels': labels}

            num_samples = len(model_features[model_names[0]]['labels'])
            inter_relation = []

            for model_name in model_names:
                feats = model_features[model_name]['feats']
                labels = model_features[model_name]['labels']
                central_feats = np.stack([np.mean(feats[labels == c], axis=0) for c in range(self.args.num_classes)])
                dists = euclidean_dist(torch.tensor(feats), torch.tensor(central_feats)).numpy()
                preds = np.argmin(dists, axis=1)
                correct = (preds == labels)
                inter_relation.append(~correct)

            inter_relation_mean = np.mean(inter_relation, axis=0)
            labels_all = model_features[model_names[0]]['labels']
            selection_result = []

            for c in range(self.args.num_classes):
                idx = np.where(labels_all == c)[0]
                inter_mean_c = inter_relation_mean[idx]
                feats_list = [model_features[m]['feats'][idx] for m in model_names]
                ranks_np = get_distance_rank(feats_list, len(model_names))
                selected = get_selected_idx(self.fraction, ranks_np, inter_mean_c)
                selection_result.extend(idx[selected])
                
            self.dst_train.transform = self.transform
            selection_result = np.array(selection_result)
        else:
            raise RuntimeError("Only balanced selection is currently implemented.")
            
        self.finish_selection(model_names)
        
        return {"indices": selection_result}