import logging
from typing import Optional
import os 
import torch
import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from uace.metrics import AccuracyMetric
from uace.utils import plot_features,plot_confusion_matrix
import numpy as np
class Evaluator:
    """Model evaluator

    Args:
        model: model to be evaluated
        device: device on which to evaluate model
        loader: dataloader on which to evaluate model
        checkpoint_path: path to model checkpoint

    """

    def __init__(
        self,
        dataset_name: 'mnist',
        model: torch.nn.Module,
        device: torch.device,
        loader: DataLoader,
        save_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        # Logging
        self.logger = logging.getLogger()

        # Device
        self.device = device

        self.dataset_name = dataset_name

        #print("----",self.dataset_name)

        # Data
        self.loader = loader

        # Model
        self.model = model

        # Save Path
        self.save_path = save_path

        #print("model",model)

        if checkpoint_path:
            self._load_from_checkpoint(checkpoint_path)

        # Metrics
        self.acc_metric = AccuracyMetric(k=1)

        self.checkpoint_path = checkpoint_path

    def evaluate(self) -> float:
        """Evaluates the model

        Returns:
            (float) accuracy (on a 0 to 1 scale)

        """
        #plot
        all_features, all_labels = [], []

        original_labels    = np.zeros(len(self.loader.dataset),dtype=np.int32)
        predicted_labels = np.zeros(len(self.loader.dataset),dtype=np.int32)


        # Progress bar
        pbar = tqdm.tqdm(total=len(self.loader), leave=False)
        pbar.set_description("Evaluating... ")

        # Set to eval
        self.model.eval()

        # Loop
        for data, target,true_target,indexs in self.loader:
            with torch.no_grad():
                # To device
                data, target,true_target,indexs = data.to(self.device), target.to(self.device),true_target.to(self.device),indexs.to(self.device)

                # Forward
                # out = self.model(data)
                # features = out
                # 
                if self.dataset_name == 'fashionmnist' or self.dataset_name == 'fashionmnist':
                    out = self.model(data)
                    features = out
                else:
                    out = self.model(data)
                    features = out

                self.acc_metric.update(out, target)

                # Update progress bar
                pbar.update()

                #plot
                all_features.append(features.data.cpu().numpy())
                all_labels.append(true_target.data.cpu().numpy())

                original_labels[indexs.cpu().detach().numpy().tolist()] = torch.argmax(F.softmax(out,dim=1),dim=1).cpu().detach().numpy().tolist()
                predicted_labels[indexs.cpu().detach().numpy().tolist()] = target.cpu().detach().numpy().tolist()

            
        pbar.close()

        accuracy = self.acc_metric.compute()
        self.logger.info(f"Accuracy: {accuracy:.4f}\n")

        print("--------",np.sum(original_labels==predicted_labels)/len(predicted_labels))



        num_classes = 10 
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        #print("---------------------------------------",self.checkpoint_path)
        #print("--",os.path.split(self.checkpoint_path)[0])
        if self.save_path != None:
            save_dir = self.save_path
        else:
            save_dir = os.path.split(self.checkpoint_path)[0]

        plot_features(all_features, all_labels, num_classes, epoch=1, prefix='Test',save_dir=save_dir)
        plot_confusion_matrix(y_true=original_labels,
                                  y_pred=predicted_labels,
                                  dataset_name=self.dataset_name,
                                  normalize=True,
                                  prefix='matrix',
                                  save_dir=save_dir)
    
        return accuracy

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
