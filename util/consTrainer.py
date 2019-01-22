#!coding:utf-8
import torch
from torch.nn import functional as F
import time, datetime

from pathlib import Path
from util.datasets import NO_LABEL
from util.ramps import exp_rampup

class Trainer:

    def __init__(self, model, ema_model, optimizer, loss_fn, device, config, writer=None, save_dir=None, save_freq=5):
        self.model = model
        self.ema_model = ema_model
        self.model_type = config.model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_dir = '{}{}_{}_{}_{}'.format(config.save_dir,
                                               config.arch,
                                               config.dataset,
                                               config.model,
                                               datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))
        self.save_freq = config.save_freq
        self.device = device
        self.writer = writer
        self.labeled_bs = config.labeled_batch_size
        self.global_step = 0
        self.epoch = 0
        self.weight = config.cons_weight
        self.rampup = exp_rampup(config.weight_rampup)
        self.ema_decay = config.ema_decay
        self.cons_loss = F.mse_loss
        
    def _iteration(self, data_loader, print_freq, is_train=True):
        loop_loss = []
        accuracy, accuracy_ema = [],[]
        labeled_n = 0
        mode = "train" if is_train else "test"
        for batch_idx, (data, targets) in enumerate(data_loader):
            self.global_step += 1
            assert len(data)==2
            data, data1 = data[0].to(self.device), data[1].to(self.device) 
            targets = targets.to(self.device)
            outputs = self.model(data)
            ## Training Phase
            if is_train:
                labeled_bs = self.labeled_bs
                labeled_loss = torch.sum(self.loss_fn(outputs, targets)) / labeled_bs
                ## consistency loss
                if self.model_type == 'ema':
                    self.update_ema(self.model, self.ema_model, self.ema_decay, self.global_step)
                with torch.no_grad():
                    ema_outputs = self.ema_model(data1)

                cons_weight = self.rampup(self.epoch)*self.weight
                # using mse_with_softmax seem better
                cons_loss = cons_weight* self.cons_loss(F.softmax(outputs,1), F.softmax(ema_outputs,1))
                #cons_loss = cons_weight)* self.cons_loss(outputs,ema_outputs)

                loss = labeled_loss + cons_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            ## Testing Phase
            else:
                labeled_bs = data.size(0)
                labeled_loss = cons_loss = torch.Tensor([0])
                loss = torch.mean(self.loss_fn(outputs, targets))
            labeled_n += labeled_bs

            loop_loss.append(loss.item() / len(data_loader))
            acc = targets.eq(outputs.max(1)[1]).sum().item()
            acc_ema = targets.eq(ema_outputs.max(1)[1]).sum().item() if is_train else 0
            accuracy.append(acc)
            accuracy_ema.append(acc_ema)
            if print_freq>0 and (batch_idx%print_freq)==0:
                print(f"[{mode}][{batch_idx:<3}]\t labeled: {labeled_loss.item():.3f}\t "\
                    f"cons: {cons_loss.item():.3f} \t "\
                    f"loss: {loss.item():.3f}\t Acc: {acc/labeled_bs:.3%}\t EMA Acc: {acc_ema/labeled_bs:.3%}")
            if self.writer:
                self.writer.add_scalar(mode+'_global_loss', loss.item(), self.global_step)
                self.writer.add_scalar(mode+'_global_accuracy', acc/labeled_bs, self.global_step)
        print(f">>>[{mode}]loss: {sum(loop_loss):.3f}\t "\
            f"Acc: {sum(accuracy)/labeled_n:.3%}\t EMA Acc: {sum(accuracy_ema)/labeled_n:.3%}")
        if self.writer:
            self.writer.add_scalar(mode+'_epoch_loss', sum(loop_loss), self.epoch)
            self.writer.add_scalar(mode+'_epoch_accuracy', sum(accuracy)/labeled_n, self.epoch)

        return loop_loss, accuracy

    def train(self, data_loader, print_freq=20):
        self.model.train()
        with torch.enable_grad():
            loss, correct = self._iteration(data_loader, print_freq)

    def test(self, data_loader, print_freq=10):
        self.model.eval()
        with torch.no_grad():
            loss, correct = self._iteration(data_loader, print_freq, is_train=False)

    def loop(self, epochs, train_data, test_data, scheduler=None, print_freq=-1):
        for ep in range(epochs):
            self.epoch = ep
            if scheduler is not None:
                scheduler.step()
            print("------ Training epochs: {} ------".format(ep))
            self.train(train_data, print_freq)
            print("------ Testing epochs: {} ------".format(ep))
            self.test(test_data, print_freq)
            ## save model
            if self.save_freq!=0 and (ep+1)%self.save_freq == 0:
                self.save(ep)

    def update_ema(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step +1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1-alpha, param.data)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch,
                    "weight": self.model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            save_target = model_out_path / "model_epoch_{}.pth".format(epoch)
            torch.save(state, save_target)
            print('==> save model to {}'.format(save_target))
