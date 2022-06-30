# A class to train models
from tqdm import tqdm

import torch
import wandb

from .models.unet import Unet
from .loss import BCEDiceLoss, DiceLoss, FocalLoss, BCELogitsLoss
from . import utils

class Trainer:
    def __init__(self, model_opts, train_par, loaders):
        if train_par.gpu_to_use == 'default':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(train_par.gpu_to_use if torch.cuda.is_available() else "cpu")

        self.set_model(model_opts)
        self.loaders = loaders
        
        #self.op = torch.optim.Adam(self.model.parameters(), lr=train_par.lr)
        #self.op = torch.optim.RMSprop(self.model.parameters(), lr=train_par.lr, alpha=0.99, eps=1e-08, weight_decay=0.2, momentum=0.9, centered=False)
        self.op = torch.optim.Adadelta(self.model.parameters(), lr=train_par.lr, rho=0.8, eps=1e-06, weight_decay=0)
        #self.op = torch.optim.SGD(self.model.parameters(), lr=train_par.lr, momentum=0.9)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.op, 'min', patience=5)

        self.eval_threshold = train_par.eval_threshold
        self.patience = train_par.patience
        self.early_stopping_flag = train_par.early_stopping_flag
        self.results_model_filename = train_par.results_model_filename
        self.train_par = train_par

    def set_model(self, model_opts):
        model_def = globals()[model_opts.name]
        self.model = model_def(**model_opts.args)
        # self.multi_cls = True if model_opts.args.outchannels > 1 else False
        wandb.watch(self.model)
        # self.model.to(self.device)
        self.model.to(self.device)

    def get_loss(self, y_hat, y):
        if self.train_par.loss_opts.name != 'default':
            self.loss_f = globals()[self.train_par.loss_opts.name]

        if self.train_par.loss_opts.name == 'BCELogitsLoss':
            if self.train_par.loss_opts.args.weight == 'default':
                return self.loss_f(y_hat, y)
            else:
                return self.loss_f(y_hat, y, weight = self.pos_weights)

        if self.train_par.loss_opts.name == 'FocalLoss':
            return self.loss_f(y_hat, y)
        
        if self.train_par.loss_opts.name == 'BCEDiceLoss':
            if self.train_par.loss_opts.args.weight == 'default':
                return self.loss_f(y_hat, y, device=self.device)
            else:
                return self.loss_f(y_hat, y, weight = self.train_par.loss_opts.args.weight, device=self.device)
        
        if self.train_par.loss_opts.name == 'DiceLoss':
            return self.loss_f(y_hat, y)        

    def train_epoch(self, train_loader):
        self.model.train
        total_loss = 0
        for img, mask in tqdm(train_loader):
            self.pos_weights = utils.pos_weight_batch(mask)
            img, mask = img.to(self.device, dtype=torch.float), mask.to(self.device, dtype=torch.float)
            pred = self.model(img)
            loss = self.get_loss(pred, mask)
            total_loss += loss.item()
            loss.backward()
            self.op.step()
            self.op.zero_grad()

        return total_loss / len(train_loader)
    
    def validation(self, dev_loader):
        self.model.eval()  
        total_loss = 0
        dice_score = 0
        total_tp , total_fp, total_tn, total_fn = 0, 0, 0, 0
        total_precision, total_recall, total_accuracy, total_f1 = 0, 0, 0, 0

        for img, mask in tqdm(dev_loader):
            self.pos_weights = utils.pos_weight_batch(mask)
            img, mask = img.to(self.device, dtype=torch.float), mask.to(self.device, dtype=torch.float)
            with torch.no_grad():
                pred_mask = self.model(img)
                loss = self.get_loss(pred_mask, mask)
                total_loss += loss.item()
                self.op.zero_grad()

                pred = torch.sigmoid(pred_mask)
                pred = (pred > self.eval_threshold).float()
                dice, _ = utils.dice_coeff_batch(pred, mask, device=self.device)
                dice_score += dice.item()

                tp, fp, tn, fn, precision, recall, accuracy, f1 = utils.confusion_matrix(pred, mask)
                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn

                total_precision += precision
                total_recall += recall
                total_accuracy += accuracy
                total_f1 += f1

        return total_loss/len(dev_loader), dice_score/len(dev_loader), total_tp/len(dev_loader), \
                total_fp/len(dev_loader), total_tn/len(dev_loader), total_fn/len(dev_loader), total_precision/len(dev_loader), \
                total_recall/len(dev_loader), total_accuracy/len(dev_loader), total_f1/len(dev_loader)

    def train(self, epochs):
        early_stopping = utils.EarlyStopping(patience=self.patience, verbose=True, path=self.results_model_filename)
        for epoch in range(epochs):
            train_loss = self.train_epoch(self.loaders['train'])
            dev_loss, dev_dice, dev_tp, dev_fp, dev_tn, dev_fn, dev_precision, dev_recall, dev_accuracy, dev_f1 = self.validation(self.loaders['dev'])
            print(f'Epoch {epoch}/{epochs}: training loss = {train_loss}, dev loss = {dev_loss}, dev dice = {dev_dice}, dev precision = {dev_precision}, dev recall = {dev_recall}, dev accuracy = {dev_accuracy}, dev f1 = {dev_f1}')
            wandb.log({"loss/train": train_loss, "loss/dev": dev_loss, "dev_metrics/dice": dev_dice, "dev_metrics/f1": dev_f1, \
                        "dev_metrics/precision": dev_precision, "dev_metrics/recall": dev_recall, "dev_metrics/accuracy": dev_accuracy, \
                        "dev_metrics/tp": dev_tp, "dev_metrics/fp": dev_fp, "dev_metrics/tn": dev_tn, "dev_metrics/fn": dev_fn}, step=epoch)

            self.scheduler.step(dev_loss)
            # Adding early stopping according to the evolution of the validation loss
            if self.early_stopping_flag:
                early_stopping(dev_loss, self.model)
                if early_stopping.counter == 0: wandb.log({"dev_metrics/best_score":early_stopping.best_score}, step=epoch)
                if early_stopping.early_stop:
                    print(f'Early stopping')
                    break

            self.model.load_state_dict(torch.load(self.results_model_filename))

    def predict(self):
        test_loader = self.loaders['test']
        L = len(test_loader)

        dice_score = 0
        total_tp , total_fp, total_tn, total_fn = 0, 0, 0, 0
        total_precision, total_recall, total_accuracy, total_f1 = 0, 0, 0, 0

        for img, mask in tqdm(test_loader):
            img, mask = img.to(self.device, dtype=torch.float), mask.to(self.device, dtype=torch.float)
            with torch.no_grad():
                pred_mask = self.model(img)
                #loss = self.loss_f(pred_mask, mask)
                #total_loss += loss
                #self.op.zero_grad()

                pred = torch.sigmoid(pred_mask)
                pred = (pred > self.eval_threshold).float()
                dice, _ = utils.dice_coeff_batch(pred, mask, device=self.device)
                dice_score += dice.item()

                tp, fp, tn, fn, precision, recall, accuracy, f1 = utils.confusion_matrix(pred, mask)
                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn

                total_precision += precision
                total_recall += recall
                total_accuracy += accuracy
                total_f1 += f1

        test_dice, test_tp, test_fp, test_tn, test_fn, test_precision, test_recall, test_accuracy, test_f1 = dice_score/L, total_tp/L, \
                total_fp/L, total_tn/L, total_fn/L, total_precision/L, total_recall/L, total_accuracy/L, total_f1/L
        
        print(f'test dice = {test_dice}, test precision = {test_precision}, test recall = {test_recall}, test accuracy = {test_accuracy}, test f1 = {test_f1}')
        wandb.log({"test_metrics/dice": test_dice, "test_metrics/f1": test_f1, \
                "test_metrics/precision": test_precision, "test_metrics/recall": test_recall, "test_metrics/accuracy": test_accuracy, \
                "test_metrics/tp": test_tp, "test_metrics/fp": test_fp, "test_metrics/tn": test_tn, "test_metrics/fn": test_fn})

