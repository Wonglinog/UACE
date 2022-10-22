import logging
import os
import time
from typing import Optional
import numpy as np
import torch
import os 
import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F 

from uace.metrics import AccuracyMetric, LossMetric
from uace.utils import plot_features,plot_hist,plot_loss
import copy


class Trainer:
    """Model trainer

    Args:
        model: model to train
        loss_fn: loss function
        optimizer: model optimizer
        epochs: number of epochs
        device: device to train the model on
        train_loader: training dataloader
        val_loader: validation dataloader
        scheduler: learning rate scheduler
        update_sched_on_iter: whether to call the scheduler every iter or every epoch
        grad_clip_max_norm: gradient clipping max norm (disabled if None)
        writer: writer which logs metrics to TensorBoard (disabled if None)
        save_path: folder in which to save models (disabled if None)
        checkpoint_path: path to model checkpoint, to resume training

    """

    def __init__(
        self,
        dataset_name: 'mnist',
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional = None,  # Type: torch.optim.lr_scheduler._LRScheduler
        update_sched_on_iter: bool = False,
        grad_clip_max_norm: Optional[float] = None,
        writer: Optional[SummaryWriter] = None,
        save_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        mixed_precision: bool = False,
    ) -> None:

        self.dataset_name = dataset_name
        # Logging
        self.logger = logging.getLogger()
        self.writer = writer

        # Saving
        self.save_path = save_path

        # Device
        self.device = device

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Model
        self.model = model

        #print("model",model)

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.update_sched_on_iter = update_sched_on_iter
        self.grad_clip_max_norm = grad_clip_max_norm
        self.epochs = epochs
        self.start_epoch = 0

        # Floating-point precision
        self.mixed_precision = (
            True if self.device.type == "cuda" and mixed_precision else False
        )
        self.scaler = GradScaler() if self.mixed_precision else None

        if checkpoint_path:
            self._load_from_checkpoint(checkpoint_path)

        # Metrics
        self.train_loss_metric = LossMetric()
        self.train_acc_metric = AccuracyMetric(k=1)
        self.train_acc_true_metric = AccuracyMetric(k=1)
        self.train_acc_noise_metric = AccuracyMetric(k=1)

        self.val_loss_metric = LossMetric()
        self.val_acc_metric = AccuracyMetric(k=1)

        self.results_last_ten =0


        self.data_clean_grad = []
        self.data_noise_grad = []

        #best test acc
        
        self.best_val_acc_metric = 0

    def train(self) -> None:
        """Trains the model"""
        self.logger.info("Beginning training")
        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            start_epoch_time = time.time()
            if self.mixed_precision:
                self._train_loop_amp(epoch)
            else:
                self._train_loop(epoch)

            if self.val_loader is not None:
                self._val_loop(epoch)

            epoch_time = time.time() - start_epoch_time
            self._end_loop(epoch, epoch_time)

        train_time_h = (time.time() - start_time) / 3600
        self.logger.info(f"Finished training! Total time: {train_time_h:.2f}h")
        self._save_model(os.path.join(self.save_path, "final_model.pt"), self.epochs)

    def _train_loop(self, epoch: int) -> None:
        """
        Regular train loop

        Args:
            epoch: current epoch
        """
        # Progress bar
        pbar = tqdm.tqdm(total=len(self.train_loader), leave=False)
        pbar.set_description(f"Epoch {epoch} | Train")

        # Set to train
        self.model.train()


        # Loop
        for data, target,true_target,indexs in self.train_loader:
            # To device
            data, target,true_target,indexs = data.to(self.device), target.to(self.device),true_target.to(self.device),indexs.to(self.device)

            # Forward + backward
            self.optimizer.zero_grad()
            if self.dataset_name == 'mnist' or self.dataset_name == 'fashionmnist':
                #out,features = self.model(data)
                out = self.model(data)
                features = out
            else:
                out = self.model(data)
                features = out
            loss,loss_instance = self.loss_fn(out, target)
            loss.backward()

            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )

            self.optimizer.step()

            # Update scheduler if it is iter-based
            if self.scheduler is not None and self.update_sched_on_iter:
                self.scheduler.step()

            # Update metrics
            self.train_loss_metric.update(loss.item(), data.shape[0])
            self.train_acc_metric.update(out, target)

            self.train_acc_true_metric.update(out, true_target)

            target_noise = target.clone()
            target_noise[target_noise==true_target]=-1
            self.train_acc_noise_metric.update(out, target_noise)


            # Update progress bar
            pbar.update()
            pbar.set_postfix_str(f"Loss: {loss.item():.3f}", refresh=False)



        # Update scheduler if it is epoch-based
        if self.scheduler is not None and not self.update_sched_on_iter:
            self.scheduler.step()

        pbar.close()

    def _train_loop_amp(self, epoch: int) -> None:
        """
        Train loop with Automatic Mixed Precision

        Args:
            epoch: current epoch
        """
        plot = True 
        # plot 
        if plot:
            all_features, all_labels = [], []

            if self.dataset_name == 'mnist' or self.dataset_name == 'fashionmnist' :
                num_classes =10
            elif self.dataset_name == 'cifar10' or self.dataset_name =="animal10n":
                num_classes =10
            elif self.dataset_name == 'cifar100':
                num_classes =100
            elif self.dataset_name == 'clothing1m':
                num_classes =14
            else:
                assert 1 == 0

           #print("------------",self.train_loader.dataset.data[0]
            results         = np.zeros((len(self.train_loader.dataset), num_classes), dtype=np.float32) 
            loss_results    = np.zeros(len(self.train_loader.dataset), dtype=np.float32) 
            train_labels    = np.zeros(len(self.train_loader.dataset),dtype=np.int32)
            train_labels_gt = np.zeros(len(self.train_loader.dataset),dtype=np.int32)


            


        # Progress bar
        pbar = tqdm.tqdm(total=len(self.train_loader), leave=False)
        pbar.set_description(f"Epoch {epoch} | Train")

        # Set to train
        self.model.train()



        # Loop
        for data, target,true_target,indexs in self.train_loader:

            #print(data.shape, target, target, indexs)
            # To device
            data, target,true_target,indexs = data.to(self.device), target.to(self.device),true_target.to(self.device),indexs.to(self.device)

            # Forward + backward
            self.optimizer.zero_grad()

            # Use amp in forward pass
            with autocast():
                if self.dataset_name == 'mnist' or self.dataset_name == 'fashionmnist':
                    out = self.model(data)
                    features = out
                else:
                    out = self.model(data)
                    features = out
                loss,loss_instance = self.loss_fn(out, target)

                pred = F.softmax(out,dim=1)
                #print("++++++++++++++",pred.size)

                results[indexs.cpu().detach().numpy().tolist()] = pred.cpu().detach().numpy().tolist()
                loss_results[indexs.cpu().detach().numpy().tolist()] = loss_instance.cpu().detach().numpy().tolist()
                train_labels[indexs.cpu().detach().numpy().tolist()] = target.cpu().detach().numpy().tolist()
                train_labels_gt[indexs.cpu().detach().numpy().tolist()] = true_target.cpu().detach().numpy().tolist()

            # Backward pass with scaler
            self.scaler.scale(loss).backward()

            # Unscale before gradient clipping
            self.scaler.unscale_(self.optimizer)

            if self.grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_max_norm
                )

            # Update optimizer and scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update scheduler if it is iter-based
            if self.scheduler is not None and self.update_sched_on_iter:
                self.scheduler.step()

            # Update metrics
            self.train_loss_metric.update(loss.item(), data.shape[0])
            self.train_acc_metric.update(out, target)


            self.train_acc_true_metric.update(out, true_target)

            target_noise = target.clone()
            target_noise[target_noise==true_target]=-1
            self.train_acc_noise_metric.update(out, target_noise)


            # Update progress bar
            pbar.update()
            pbar.set_postfix_str(f"Loss: {loss.item():.3f}", refresh=False)

            #plot
            if plot:
                all_features.append(features.data.cpu().numpy())
                all_labels.append(true_target.data.cpu().numpy())

        # Update scheduler if it is epoch-based
        if self.scheduler is not None and not self.update_sched_on_iter:
            self.scheduler.step()

        pbar.close()

        # plot 
        if plot:
            num_classes = 10 
            all_features = np.concatenate(all_features, 0)
            all_labels = np.concatenate(all_labels, 0)
            #print("---------------------------------------",self.save_path)
            plot_features(all_features, all_labels, num_classes, epoch, prefix='train',save_dir=self.save_path)

            #inds_noisy      = np.asarray([ind for ind in range(len(self.train_loader.dataset)) if train_labels[ind] != train_labels_gt[ind]])

            #print("*****",train_labels == train_labels_gt)
            if np.sum(train_labels-train_labels_gt) !=0 :

                #print("----------true-------------")
                inds_noisy      = np.asarray([ind for ind in range(len(self.train_loader.dataset)) if train_labels[ind] != train_labels_gt[ind]])
                inds_clean      = np.delete(np.arange(len(self.train_loader.dataset)), inds_noisy)
                data_clean_pred = results[inds_clean, train_labels[inds_clean]]
                data_noise_pred = results[inds_noisy, train_labels[inds_noisy]]
                plot_hist(data_clean_pred,data_noise_pred,epoch,prefix='hist',save_dir=self.save_path)

                loss_clean = loss_results[inds_clean]
                loss_noise = loss_results[inds_noisy]

                #print(loss_clean,loss_noise)
                self.data_clean_grad.append(np.mean(loss_clean).tolist())
                self.data_noise_grad.append(np.mean(loss_noise).tolist())

                #print(self.data_clean_grad,self.data_noise_grad)

        plot_loss(self.data_clean_grad,self.data_noise_grad,epoch,self.epochs,prefix='grad',save_dir=self.save_path)



            #else:
                #print("---------False-------")


    def _val_loop(self, epoch: int) -> None:
        """
        Standard validation loop

        Args:
            epoch: current epoch
        """
        # Progress bar
        pbar = tqdm.tqdm(total=len(self.val_loader), leave=False)
        pbar.set_description(f"Epoch {epoch} | Validation")

        # Set to eval
        self.model.eval()

        # Loop
        for data, target,true_target,indexs in self.val_loader:
            with torch.no_grad():
                # To device
                data, target,true_target,indexs = data.to(self.device), target.to(self.device),true_target.to(self.device),indexs.to(self.device)

                # Forward
                if self.dataset_name == 'mnist' or self.dataset_name == 'fashionmnist':
                    out = self.model(data)
                    features = out
                else:
                    out = self.model(data)
                    features = out
                # out = self.model(data)
                loss,_ = self.loss_fn(out, target)

                # Update metrics
                self.val_loss_metric.update(loss.item(), data.shape[0])
                self.val_acc_metric.update(out, target)

                # Update progress bar
                pbar.update()
                pbar.set_postfix_str(f"Loss: {loss.item():.3f}", refresh=False)
        self.best_val_acc_metric = max(self.val_acc_metric.compute(),self.best_val_acc_metric)
        pbar.close()

        if epoch >=190:
            self.results_last_ten+=self.val_acc_metric.compute()
        else:
            self.results_last_ten = 0

        if epoch ==200:
            self.results_last_ten = float(self.results_last_ten/10)
            self.logger.info(f"Last_Ten_Accuracy: {self.results_last_ten:.4f}\n")

    def _end_loop(self, epoch: int, epoch_time: float):
        # Print epoch results
        self.logger.info(self._epoch_str(epoch, epoch_time))

        # Write to tensorboard
        if self.writer is not None:
            self._write_to_tb(epoch)

        # Save model
        if self.save_path is not None:
            self._save_model(os.path.join(self.save_path, "most_recent.pt"), epoch)

        # Clear metrics
        self.train_loss_metric.reset()
        self.train_acc_metric.reset()
        self.train_acc_true_metric.reset()
        self.train_acc_noise_metric.reset()

        if self.val_loader is not None:
            self.val_loss_metric.reset()
            self.val_acc_metric.reset()

    def _epoch_str(self, epoch: int, epoch_time: float):
        s = f"Epoch {epoch} "
        s += f"| Train loss: {self.train_loss_metric.compute():.4f} "
        s += f"| Train acc: {self.train_acc_metric.compute():.4f} "
        s += f"| Train accT: {self.train_acc_true_metric.compute():.4f} "
        s += f"| Train accN: {self.train_acc_noise_metric.compute():.4f} "
        s += f"| Train accO: {1-self.train_acc_true_metric.compute()-self.train_acc_noise_metric.compute():.4f} "
        if self.val_loader is not None:
            s += f"| Val loss: {self.val_loss_metric.compute():.4f} "
            s += f"| Val acc: {self.val_acc_metric.compute():.4f} "
            s += f"| Best Val acc: {self.best_val_acc_metric:.4f} "
        s += f"| Epoch time: {epoch_time:.1f}s"

        return s

    def _write_to_tb(self, epoch):
        self.writer.add_scalar("Loss/train", self.train_loss_metric.compute(), epoch)
        self.writer.add_scalar("Accuracy/train", self.train_acc_metric.compute(), epoch)

        if self.val_loader is not None:
            self.writer.add_scalar("Loss/val", self.val_loss_metric.compute(), epoch)
            self.writer.add_scalar("Accuracy/val", self.val_acc_metric.compute(), epoch)

    def _save_model(self, path, epoch):
        obj = {
            "epoch": epoch + 1,
            "optimizer": self.optimizer.state_dict(),
            "model": self.model.state_dict(),
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "scaler": self.scaler.state_dict() if self.mixed_precision else None,
        }
        torch.save(obj, os.path.join(self.save_path, path))

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.start_epoch = checkpoint["epoch"]

        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if self.mixed_precision and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scheduler"])

        if self.start_epoch > self.epochs:
            raise ValueError("Starting epoch is larger than total epochs")

        self.logger.info(f"Checkpoint loaded, resuming from epoch {self.start_epoch}")
