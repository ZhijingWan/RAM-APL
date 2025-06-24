from .earlytrain import EarlyTrain
import torch
import numpy as np
from .methods_utils import euclidean_dist
from ..nets.nets_utils import MyDataParallel
from datetime import datetime

def get_median(features):
    # get the median feature vector of each class
    prot = np.median(features, axis=0, keepdims=False)
    return prot


def get_distance(features):
    prots = get_median(features)
    distance = np.linalg.norm(features - prots, axis=1)
    return distance

def get_selected_idx(sampling_rate, distance):
    budget = int(distance.shape[0]*sampling_rate)
    sorted_idx = np.argsort(distance)
    low_idx = round(0.5 * (distance.shape[0] - budget))
    high_idx = budget + low_idx
    ids = sorted_idx[low_idx:high_idx]
    return ids

class Moderate_DS(EarlyTrain):
    """
    Moderate_DS: Select samples with moderate distance to class center (not too close or too far).
    """
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

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def old_construct_matrix(self, index=None):
        self.model.eval()
        self.model.no_grad = True
        with torch.no_grad():
            with self.model.embedding_recorder:
                sample_num = self.n_train if index is None else len(index)
                matrix = torch.zeros([sample_num, self.emb_dim], requires_grad=False).to(self.args.device)

                data_loader = torch.utils.data.DataLoader(self.dst_train if index is None else
                                        torch.utils.data.Subset(self.dst_train, index),
                                                batch_size=self.args.selection_batch,
                                                num_workers=self.args.workers)

                for i, (inputs, _) in enumerate(data_loader):
                    self.model(inputs.to(self.args.device))
                    matrix[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch,
                                                             sample_num)] = self.model.embedding_recorder.embedding

        self.model.no_grad = False
        return matrix

    def construct_matrix(self, index=None):
        data_loader = torch.utils.data.DataLoader(self.dst_train if index is None else
                                        torch.utils.data.Subset(self.dst_train, index),
                                        batch_size=self.args.selection_batch,
                                        num_workers=self.args.workers)
        
        self.model.eval()
        features = []
        with torch.no_grad():
            for inputs, _ in data_loader:
                if self.model_name == 'CLIP':
                    features_batch = self.model.encode_image(inputs.to(self.args.device))
                elif self.model_name == 'DINOV2':
                    outputs = self.model(inputs.to(self.args.device))
                    features_batch = outputs.pooler_output
                elif self.model_name == 'SIGLIP':
                    inputs = inputs['pixel_values'][0].to(torch.float16)
                    outputs = self.model(inputs.to(self.args.device))
                    features_batch = outputs.pooler_output
                elif self.model_name == 'EVA-CLIP':
                    inputs = inputs['pixel_values'][0].to(torch.float16)
                    features_batch = self.model.encode_image(inputs.to(self.args.device))
                else:
                    with self.model.embedding_recorder:
                        self.model(inputs.to(self.args.device))
                        features_batch = self.model.embedding_recorder.embedding

                features.append(features_batch.float())

        return torch.cat(features, dim=0)

    def before_run(self):
        pass

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module
    
    def finish_selection(self):
        del self.model
        torch.cuda.empty_cache()

    def select(self, **kwargs):
        self.run()
        print('end training, start MDS selection at : {}'.format(datetime.now()))

        if self.balance:
            selection_result = np.array([], dtype=np.int32)
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                features = self.construct_matrix(class_index).detach().cpu().numpy()
                distance = get_distance(features)
                ids = get_selected_idx(self.fraction, distance)
                selection_result_new = class_index[ids]
                selection_result = np.append(selection_result, selection_result_new)
        else:
            raise RuntimeError("Only balanced class-wise selection is implemented in MDS.")

        self.dst_train.transform = self.transform
        self.finish_selection()

        return {"indices": selection_result}
