import torch
import numpy as np

def dice_coeff_batch(batch_bn_mask, batch_true_bn_mask, device = 'cuda'):
    """ dice_coeff_batch : function that returns the mean dice coeff for a batch of pairs 
    mask, ground truth mask """
    
    def single_dice_coeff(input_bn_mask, true_bn_mask):
        """single_dice_coeff : function that returns the dice coeff for one pair 
        of mask and ground truth mask"""

        # The eps value is used for numerical stability
        eps = 0.0001

        # Computing intersection and union masks
        inter_mask = torch.dot(input_bn_mask.view(-1), true_bn_mask.view(-1))
        union_mask = torch.sum(input_bn_mask) + torch.sum(true_bn_mask) + eps

        # Computing the Dice coefficient
        return (2 * inter_mask.float() + eps) / union_mask.float()

    # Init dice score for batch (GPU ready)
    if batch_bn_mask.is_cuda: dice_score = torch.FloatTensor(1).cuda(device=device).zero_()
    else: dice_score = torch.FloatTensor(1).zero_()

    # Compute Dice coefficient for the given batch
    for pair_idx, inputs in enumerate(zip(batch_bn_mask, batch_true_bn_mask)):
        dice_score +=  single_dice_coeff(inputs[0], inputs[1])
    
    # Return the mean Dice coefficient over the given batch
    dice_batch = dice_score / (pair_idx + 1)

    return dice_batch, 1 - dice_batch

def metrics(p_n, tp, fp, tn, fn):
    """ Returns accuracy, precision, recall, f1 based on the inputs 
    tp : true positives, fp: false positives, tn: true negatives, fn: false negatives
    For details please check : https://en.wikipedia.org/wiki/Precision_and_recall
    """
    try:
        # Computing the accuracy
        accuracy  = (tp + tn) / p_n

        # Computing the precision
        precision =  tp / (tp + fp)

        # Computing the recall
        recall    =  tp / (tp + fn)

        # Computing the f1
        f1        =  2 * tp / (2 * tp + fp + fn)
    
    except ZeroDivisionError:
        precision, recall, accuracy, f1 = 0, 0, 0, 0 
    
    return precision, recall, accuracy, f1

def confusion_matrix(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """
    # Computing the confusion vector
    confusion_vector = prediction / truth
    
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)
    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item()

    # Computing the total (p+n)
    p_n = tp + fp + tn + fn

    # Computing the precision, recall, accuracy, f1 metrics
    precision, recall, accuracy, f1 = metrics(p_n, tp, fp, tn, fn)

    return tp/p_n, fp/p_n, tn/p_n, fn/p_n, precision, recall, accuracy, f1 

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def pos_weight_batch(mask):
    size = mask.size()
    pos = torch.sum(mask)
    total_px = (size[-1]**2) * size[0]
    # print(size, total_px, pos)
    return (total_px - pos) / pos # neg / pos
