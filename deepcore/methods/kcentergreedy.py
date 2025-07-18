from .earlytrain import EarlyTrain
import torch
import numpy as np
from .methods_utils import euclidean_dist
from ..nets.nets_utils import MyDataParallel
from datetime import datetime

def k_center_greedy(matrix, budget: int, metric, device, random_seed=None, index=None, already_selected=None,
                    print_freq: int = 20):
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
    elif type(matrix) == np.ndarray:
        assert matrix.ndim == 2
        matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)
    else:
        raise TypeError("Input matrix must be numpy array or torch tensor.")

    sample_num = matrix.shape[0]
    assert sample_num >= 1

    if budget < 0:
        raise ValueError("Illegal budget size.")
    elif budget > sample_num:
        budget = sample_num

    if index is not None:
        assert matrix.shape[0] == len(index)
    else:
        index = np.arange(sample_num)

    assert callable(metric)

    already_selected = np.array(already_selected)

    with torch.no_grad():
        np.random.seed(random_seed)
        if already_selected.__len__() == 0:
            select_result = np.zeros(sample_num, dtype=bool)
            # Randomly select one initial point.
            already_selected = [np.random.randint(0, sample_num)]
            budget -= 1
            select_result[already_selected] = True
        else:
            select_result = np.in1d(index, already_selected)

        num_of_already_selected = np.sum(select_result)

        # Initialize a (num_of_already_selected+budget-1)*sample_num matrix storing distances of pool points from
        # each clustering center.
        dis_matrix = -1 * torch.ones([num_of_already_selected + budget - 1, sample_num], requires_grad=False).to(device)

        dis_matrix[:num_of_already_selected, ~select_result] = metric(matrix[select_result], matrix[~select_result])

        if (num_of_already_selected + budget - 1):
            mins = torch.min(dis_matrix[:num_of_already_selected, :], dim=0).values

        for i in range(budget):
            if i % print_freq == 0:
                print("| Selecting [%3d/%3d]" % (i + 1, budget))
            p = torch.argmax(mins).item()
            select_result[p] = True

            if i == budget - 1:
                break
            mins[p] = -1
            dis_matrix[num_of_already_selected + i, ~select_result] = metric(matrix[[p]], matrix[~select_result])
            mins = torch.min(mins, dis_matrix[num_of_already_selected + i])
    return index[select_result]


class kCenterGreedy(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=0,
                 specific_model="ResNet18", balance: bool = False, already_selected=[], metric="euclidean",
                 torchvision_pretrain: bool = False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs=epochs, specific_model=specific_model,
                         torchvision_pretrain=torchvision_pretrain, **kwargs)

        if already_selected.__len__() != 0:
            if min(already_selected) < 0 or max(already_selected) >= self.n_train:
                raise ValueError("List of already selected points out of the boundary.")
        self.already_selected = np.array(already_selected)
        self.balance = balance
        self.transform = self.dst_train.transform

        if metric == "euclidean":
            self.metric = euclidean_dist
        elif callable(metric):
            self.metric = metric
        else:
            self.metric = euclidean_dist
            self.run = lambda : self.finish_run()
            def _construct_matrix(index=None):
                data_loader = torch.utils.data.DataLoader(
                    self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                    batch_size=self.n_train if index is None else len(index),
                    num_workers=self.args.workers)
                inputs, _ = next(iter(data_loader))
                return inputs.flatten(1).requires_grad_(False).to(self.args.device)
            self.construct_matrix = _construct_matrix

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

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
        print('end training, start KCG selection at : {}'.format(datetime.now()))
        if self.balance:
            selection_result = np.array([], dtype=np.int32)
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                features = self.construct_matrix(class_index).detach().cpu().numpy()
                selection_result_new = k_center_greedy(features,
                                                        budget=round(
                                                            self.fraction * len(class_index)),
                                                        metric=self.metric,
                                                        device=self.args.device,
                                                        random_seed=self.random_seed,
                                                        index=class_index,
                                                        already_selected=self.already_selected[
                                                            np.in1d(self.already_selected,
                                                                    class_index)],
                                                        print_freq=self.args.print_freq)

                selection_result = np.append(selection_result, selection_result_new)
        else:
            matrix = self.construct_matrix()
            selection_result = k_center_greedy(matrix, budget=self.coreset_size,
                                               metric=self.metric, device=self.args.device,
                                               random_seed=self.random_seed,
                                               already_selected=self.already_selected, print_freq=self.args.print_freq)
        
        self.dst_train.transform = self.transform
        self.finish_selection()
        
        return {"indices": selection_result}