#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:34:22 2022
    Sorbonne université, Sorbonne Center for Artificial Intelligence - SCAI
    Institut du Cerveau - Paris Brain Institute - ICM (INSERM, CNRS, Sorbonne Univeristé, AP-HP), INRIA "ARAMIS Lab"
@author: mehdi.ounissi
@email : mehdi.ounissi@icm-institue.org
         mehdi.ounissi@etu.sorbonne-universite.fr
"""
import sys
sys.path.append('../../')

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils import CustomDataset, confusion_matrix, dice_coeff_batch, DiceLoss
from natsort import natsorted
from glob import glob
from unet import AttU_Net
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import os, sys
import random
import torch
import time

def train_pytorch_model():
    # Preparing the tensorboard to store training logs
    writer = SummaryWriter(comment='_'+fold_str+'_'+exp_name)

    # Loging the information about the current training
    logging.info(f'''[INFO] Starting training:
        Experiment name                  : {exp_name}
        Epochs number                    : {n_epoch}
        Early stop val loss- wait epochs : {wait_epochs}
        Batch size                       : {batch_size}
        Learning rate                    : {learning_rate}
        Training dataset size            : {len(train_dataset)}
        Validation dataset size          : {len(val_dataset)}
        PyTorch random seed              : {random_seed}
        Model input channels             : {n_input_channels}
        Model output channels            : {n_output_channels}
        Path to logs and ckps            : {path_to_logs}
        Cross-validation                 : {cross_val}
    ''')

    # Use the corresponding data type for the masks
    mask_data_type = torch.float32 if n_output_channels == 1 else torch.long

    # Init the best value of evaluation loss
    best_val_loss = 10000

    # Patience counter
    early_stop_count = 0

    # Starting the training
    for epoch in range(n_epoch):
        tic = time.time()
        # Make sure the model is in training mode
        model.train()
        
        # Init the epoch loss
        epoch_loss = 0

        # Train using batches
        for batch in tqdm(train_loader):
            # Load the image and mask
            image, true_mask = batch['image'], batch['mask']

            # Make sure the data loader did prepare images properly
            assert image.shape[1] == n_input_channels, \
				f'The input image size {image.shape[1]} ' \
				f', yet the model have {n_input_channels} input channels'

            # Load the image and the mask into device memory
            image = image.to(device=device, dtype=torch.float32)
            true_mask = true_mask.to(device=device, dtype=mask_data_type)

            # zero the parameter gradients to lower the memory footprint
            optimizer.zero_grad()

            # Make the prediction on the loaded image
            pred_mask = model(image)

            # Apply sigmoid in case on MSE loss
            if mse_loss: pred_mask = torch.sigmoid(pred_mask)

            # Computing the batch loss
            if not(dice_loss_flag): batch_loss = criterion(pred_mask, true_mask)
            else:
                batch_bce_loss = bse_loss(pred_mask, true_mask)
                batch_dice_loss = DiceLoss(pred_mask, true_mask)
                batch_loss = 0.5 * batch_dice_loss + 0.5 * batch_bce_loss

            # Backward pass to change the model params
            batch_loss.backward()

            # Informing the optimizer that this batch is over
            optimizer.step()

            # Adding the batch loss to quantify the epoch loss
            epoch_loss += batch_loss.item()

            # Uncomment this to clip the gradients (can help with stability)
            # torch.nn.utils.clip_grad_value_(model.parameters(), 0.01)
        
        # Evaluation of the model
        val_loss, total_dice_coeff, total_TP, total_FP, total_TN, total_FN, total_pres, total_rec, total_acc, total_f = evaluation_pytorch_model(model, val_loader, device)

        # Getting the mean loss value
        epoch_loss = epoch_loss/len(train_loader)
        val_loss   = val_loss/len(val_loader)

        scheduler.step(val_loss)

        # Putting the model into training mode -to resume the training phase-
        model.train()
        
        # Save the epoch training loss in the tensorboard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        
        # Save the epoch validation loss & metrics in the tensorboard
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/DICE', total_dice_coeff, epoch)
        writer.add_scalar('Metrics/TP', total_TP, epoch)
        writer.add_scalar('Metrics/FP', total_FP, epoch)
        writer.add_scalar('Metrics/TN', total_TN, epoch)
        writer.add_scalar('Metrics/FN', total_FN, epoch)
        writer.add_scalar('Metrics/precision', total_pres, epoch)
        writer.add_scalar('Metrics/recall', total_rec, epoch)
        writer.add_scalar('Metrics/accuracy', total_acc, epoch)
        writer.add_scalar('Metrics/F1-score', total_f, epoch)

        hours, rem = divmod(time.time()-tic, 3600)
        minutes, seconds = divmod(rem, 60)
        logging.info(f'''[INFO] Epoch {epoch} took {int(hours)} h {int(minutes)} min {int(seconds)}:
                Mean train loss          :  {epoch_loss}
                Mean val   loss          :  {val_loss}

        ''')
        
        if not(mse_loss) :
            logging.info(f'''
                    -- Evaluation of the model --
                    Dice                :  {total_dice_coeff}

                    TP                  :  {total_TP}
                    FP                  :  {total_FP}
                    TN                  :  {total_TN}
                    FN                  :  {total_FN}

                    Precision           :  {total_pres}
                    Recall              :  {total_rec}
                    Accuracy            :  {total_acc}
                    F1-score            :  {total_f}

            ''')

        # Saving all model's checkpoints
        if save_all_models:
            # Since DataParallel is used, adapting the parameters saving
            if n_devices > 1:
                torch.save(model.module.state_dict(), os.path.join(path_to_ckpts, f'ckp_{epoch}_{total_dice_coeff}.pth'))
            
            # Saving the parameters in case of one device
            else: torch.save(model.state_dict(), os.path.join(path_to_ckpts, f'ckp_{epoch}_{total_dice_coeff}.pth'))

        # Saving the best model
        if best_val_loss > val_loss:
            # Since DataParallel is used, adapting the parameters saving
            if n_devices > 1: torch.save(model.module.state_dict(), os.path.join(path_to_ckpts, 'best_model.pth'))
            
            # Saving the parameters in case of one device
            else: torch.save(model.state_dict(), os.path.join(path_to_ckpts, 'best_model.pth'))

            logging.info(f'''
                Best epoch {epoch} :
            
            ''')
           
            # Update the best validation loss
            best_val_loss = val_loss

            # Reset patience counter
            early_stop_count  = 0
        elif early_stop_count < wait_epochs: early_stop_count += 1
        
        else :
            logging.info(f'''[INFO] Early stop at epoch {epoch} ...''')
            break

    # Close the tensorboard writer
    writer.close()


def evaluation_pytorch_model(model, data_loader, device):
    """evaluation_pytorch_model: Evaluation of a PyTorch model and returns eval loss,
     dice coeff and the elements of a confusion matrix"""
    # Putting the model in evluation mode (no gradients are needed)
    model.eval()

    # Use the corrsponding data type of the mask
    mask_data_type = torch.float32 if n_output_channels == 1 else torch.long

    # The batch number 
    n_batch = len(data_loader)

    # Init cars needed in evaluation
    total_dice_coeff, total_loss = 0, 0
    total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0
    total_pres, total_rec, total_acc, total_f = 0, 0, 0, 0

    for batch in data_loader:
        # Load the image and mask
        image, true_mask = batch['image'], batch['mask']

        # Make sure the data loader did prepare images properly
        assert image.shape[1] == n_input_channels, \
            f'The input image size {image.shape[1]} ' \
            f', yet the model have {n_input_channels} input channels'

        # Load the image and the mask into device memory
        image = image.to(device=device, dtype=torch.float32)
        true_mask = true_mask.to(device=device, dtype=mask_data_type)

        # No need to use the gradients (no backward passes -evaluation only-)
        with torch.no_grad():

            # Computing the prediction on the input image
            pred_mask = model(image)

            # Apply sigmoid in case on MSE loss
            if mse_loss: pred_mask = torch.sigmoid(pred_mask)

            # Computing the loss
            if not(dice_loss_flag): loss = criterion(pred_mask, true_mask)
            else: loss = DiceLoss(pred_mask, true_mask)
            total_loss += loss.item()

            # Getting the binary mask
            pred = torch.sigmoid(pred_mask)
            pred = (pred > eval_threshold).float()

            # Computing the Dice coefficent
            total_dice_coeff += dice_coeff_batch(pred, true_mask).item()
            
            # Computing helpful metrics 
            tp, fp, tn, fn, precision, recall, accuracy, f1  = confusion_matrix(pred, true_mask)

            # Saving tp, fp, tn, fn in order to compute the mean values at the end of the evaluation
            total_TP   += tp
            total_FP   += fp
            total_TN   += tn
            total_FN   += fn
            
            # Saving metrics in order to compute the mean values at the end of the evaluation
            total_pres += precision
            total_rec  += recall
            total_acc  += accuracy
            total_f    += f1
    
    # Computting the mean values (tp, fp, tn, fn) over all the evaluation dataset
    total_TP   = total_TP / n_batch
    total_FP   = total_FP / n_batch
    total_TN   = total_TN / n_batch
    total_FN   = total_FN / n_batch

    # Computting the mean values -metrics- over all the evaluation dataset
    total_pres = total_pres / n_batch
    total_rec  = total_rec  / n_batch
    total_acc  = total_acc  / n_batch
    total_f    = total_f    / n_batch
    total_dice_coeff = total_dice_coeff / n_batch

    return total_loss, total_dice_coeff, total_TP, total_FP, total_TN, total_FN, total_pres, total_rec, total_acc, total_f


if __name__ == '__main__':

    ################# Hyper parameters ####################################
    # The number of epochs for the training
    n_epoch = 60

    # The batch size !(limited by how many the GPU memory can take at once)!
    batch_size = 8 # batch size for one GPU

    # Leaning rate that the optimizer uses to change the model parameters
    learning_rate = 0.01

    # True : MSE loss use , False : BCE loss or CrossEntropyLoss
    mse_loss = False

    # Early stop if the val loss didn't improve after N wait_epochs
    wait_epochs = 15

    # Save the model's parameters at the end of each epoch (if not only the
    # best model will be saved according to the validation loss)
    save_all_models = False

    # Setting a random seed for reproducibility
    random_seed = 2022
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # DICE LOSS + BCEloss flag
    dice_loss_flag = True

    # Evaluation threshold (binary masks generation and Dice coeff computing)
    eval_threshold = 0.5

    ################# EXP test fold parameters  ###########################
    # Cross testing number
    cross_test = '02'
    exp_split_folder = 'experiment_002'
    ################# DATA parameters  ####################################
    # Paths to save the prepared dataset
    main_data_dir = os.path.join('..', '..', '..','dataset','256x256')

    # Reading the sub folders of the testing phase
    test_df = pd.read_csv(os.path.join(exp_split_folder, 'test_'+cross_test+'.csv'))
    test_slides = test_df.wsi.to_list()

    train_slides_folds, val_slides_folds = [], []
    # Defining the slides folders lists for training
    train_csvs = glob(os.path.join(exp_split_folder, 'train_'+cross_test+'_*.csv'))

    for csv in train_csvs:
        train_df = pd.read_csv(csv)
        tmp_train_list = train_df.wsi.to_list()
        train_slides_folds.append(tmp_train_list)

    # Defining the slides folders lists for training
    val_csvs = glob(os.path.join(exp_split_folder, 'dev_'+cross_test+'_*.csv'))
    for csv in val_csvs:
        val_df = pd.read_csv(csv)
        tmp_val_list = val_df.wsi.to_list()
        val_slides_folds.append(tmp_val_list)
 
    # Path to the reference image (for normaliztation)
    ref_image_path = os.path.join(main_data_dir, 'A1702076 - 2021-07-21 18.41.42', 'patches', 'patch_0000.png')

    # Online normalization flag
    normalize_switch = False

    if not(normalize_switch): offline_normalization = 'macenko'
    else: 'patches'
    #######################################################################


    # when cross_val = 0 -> no cross validation is used only a 80% of 
    #                       the dataset train and 20% of it for 
    #                       validation
    #      cross_val = N -> N fold cross validation : the dataset will be
    #                                                 divided by N and one 
    #                                                 dataset fraction is 
    #                                                 used for validation
    #                                                 each time.
    #                                                 (N=5 -> 5 trainings)
    # Test folder index
    cross_val = len(train_slides_folds)

    # The folds switches (True if the training is done) 
    folds_done = [False, False, False]

    # The fold name
    fold_str =''

    # Make sure the cross_val is between [2, N]
    assert 1 < cross_val <= 5, f'[ERROR] Cross-Validation must be greater then 2 and less or equal 5, and it is {cross_val}'

    # Make sure the cross_val is between [2, N]
    assert cross_val == len(folds_done), f'[ERROR] Cross-Validation switches must match but we have {cross_val} / and {len(folds_done)} switches'

    # Rescaling the images and masks
    scale_factor = 1

    # Make sure the cross_val is between [2, N]
    assert 0 < scale_factor <= 1, '[ERROR] Scale must be between ]0, 1]'

    # The experiment name to keep track of all logs
    exp_name = ''
    if cross_val !=0:
        exp_name += 'test_'+cross_test+'_'
    exp_name += 'attention_unet_'
    exp_name += 'EP_'+str(n_epoch)
    exp_name +='_ES_'+str(wait_epochs)
    exp_name +='_BS_'+str(batch_size)
    exp_name +='_LR_'+str(learning_rate)
    exp_name +='_RS_'+str(random_seed)
    #######################################################################
    

    # Path to the log file and the saved ckps if any
    path_to_logs = os.path.join('..', '..', 'experiments', exp_name)

    # Creating the experiment folder to store all logs
    os.makedirs(path_to_logs, exist_ok = True) 

    # Cerate a logger
    logging.basicConfig(filename=os.path.join(path_to_logs, 'logfile.log'), filemode='w', 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    

    ################# Computation hyper parameters ########################
    # Number of the workers (CPUs) to be used by the dataloader (HDD -> RAM -> GPU)
    n_workers = 8

    # Make this true if you have a lot of RAM to store all the training dataset in RAM
    # (This will speed up the training at the coast of huge RAM consumption)
    pin_memory = True

    # Chose the GPU CUDA devices to make the training go much faster vs CPU use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Possibility to use at least two GPUs (available)
    if torch.cuda.device_count() > 1:
        # Log with device the training will be using (at least one GPU in this case)
        logging.info(f'[INFO] Using {torch.cuda.device_count()} {device}')

        # Log the GPUs models
        for i in range(torch.cuda.device_count()):
            logging.info(f'[INFO]      {torch.cuda.get_device_name(i)}')
        
        # For faster convolutions, but more memory
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Number of devices (GPUs) in use
        n_devices = torch.cuda.device_count()
    
    # Using one GPU (available)
    elif torch.cuda.is_available():
        # Log with device the training will be using (one GPU in this case)
        logging.info(f'[INFO] Using {device}')

        # Log the GPU model
        logging.info(f'[INFO]      {torch.cuda.get_device_name(0)}')

        # For faster convolutions, but more memory
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Number of device (GPU) in use
        n_devices = 1
    
    # No GPU available, CPU is used in this case
    else:
        # Log with device the training will be using (CPU in this case)
        logging.info(f'[INFO] Using {device}')
        
        # Since CPU will be used no need to adapt the batch size
        n_devices = 1
    #######################################################################



    ################# U-NET parameters ####################################
    # The number of input images    (RGB       ->  n_input_channels=3)
    #                               (Gray      ->  n_input_channels=1)
    n_input_channels = 3

    # The number of output classes  (N classes  ->  n_output_channels = N)
    n_output_channels = 1
    #######################################################################
    
    # defining the U-Net model
    model = AttU_Net(img_ch=n_input_channels, output_ch=n_output_channels)

    # Putting the model inside the device
    model.to(device=device, dtype=torch.float32)

    # Use all the GPUs we have
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # Optimizer used for the training phase
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.8, eps=1e-06, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    if not(dice_loss_flag):
        # The loss function used 
        if n_output_channels > 1:
            if not(mse_loss): criterion = torch.nn.CrossEntropyLoss()
            else: criterion = torch.nn.MSELoss()
        else:
            if not(mse_loss): criterion = torch.nn.BCEWithLogitsLoss()
            else: criterion = torch.nn.MSELoss()
    else:
        bse_loss = torch.nn.BCEWithLogitsLoss()

    # Preparing the test image paths lists
    test_imgs_fold, test_masks_fold = [], []
    
    logging.info(f'[INFO] Test slides sub-directory:')
    # Load testing path images
    for k in range(len(test_slides)):
        tmp_test_imgs = natsorted(glob(os.path.join(main_data_dir,test_slides[k], offline_normalization,'*.png')))
        tmp_test_masks = natsorted(glob(os.path.join(main_data_dir,test_slides[k],'masks','*.png')))
        for s in range(len(tmp_test_imgs)):
            test_imgs_fold.append(tmp_test_imgs[s])
            test_masks_fold.append(tmp_test_masks[s])

        logging.info(f'                  {test_slides[k]} : with {len(tmp_test_imgs)} images and {len(tmp_test_imgs)} masks')

        
    # Cross-validation will be used
    logging.info(f'[INFO] Cross-validation in progress ...')
    # Cross validation dataset preparation
    for i in range(cross_val):
        if not(folds_done[i]):
            logging.info(f'[INFO] Starting Fold {i+1} :')
            # Fold am for logs
            fold_str = 'FOLD-'+str(i+1)
                            
            # Defining the path to the checkpoints
            path_to_ckpts = os.path.join(path_to_logs, 'ckpts', fold_str)

            # Creating the experiment folder to store all logs
            os.makedirs(path_to_ckpts, exist_ok = True) 
            
            # Retreiving the current train & validation folds
            train_dir_fold = train_slides_folds[i]
            val_dir_fold = val_slides_folds[i]
            
            # Prepare training & validation lists
            train_imgs_fold, train_masks_fold = [], []
            val_imgs_fold, val_masks_fold = [], []
            
            logging.info(f'[INFO] Validation slides sub-directory:')
            # Load validation path images
            for k in range(len(val_dir_fold)):
                tmp_val_imgs = natsorted(glob(os.path.join(main_data_dir,val_dir_fold[k], offline_normalization,'*.png')))
                tmp_val_masks = natsorted(glob(os.path.join(main_data_dir,val_dir_fold[k],'masks','*.png')))
                for s in range(len(tmp_val_imgs)):
                    val_imgs_fold.append(tmp_val_imgs[s])
                    val_masks_fold.append(tmp_val_masks[s])
                logging.info(f'                  {val_dir_fold[k]} : with {len(tmp_val_imgs)} images and {len(tmp_val_masks)} masks')

            logging.info(f'[INFO] Training slides sub-directory:')
            # Load training path images
            for k in range(len(train_dir_fold)):
                tmp_train_imgs = natsorted(glob(os.path.join(main_data_dir,train_dir_fold[k], offline_normalization,'*.png')))
                tmp_train_masks = natsorted(glob(os.path.join(main_data_dir,train_dir_fold[k],'masks','*.png')))
                for s in range(len(tmp_train_imgs)):
                    train_imgs_fold.append(tmp_train_imgs[s])
                    train_masks_fold.append(tmp_train_masks[s])
                logging.info(f'                  {train_dir_fold[k]} : with {len(tmp_train_imgs)} images and {len(tmp_train_masks)} masks')

            # Log the current fold with the corresponding details
            logging.info(f'[INFO] Fold {i+1} : with {len(train_imgs_fold)} training, {len(val_imgs_fold)} validation images')

            # Preparing the training dataloader
            train_dataset = CustomDataset(train_imgs_fold, train_masks_fold, ref_image_path, normalize=normalize_switch,cached_data=pin_memory, n_channels=n_input_channels,scale=scale_factor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size*n_devices, shuffle=normalize_switch, pin_memory=pin_memory, num_workers=n_workers)

            # Preparing the validation dataloader
            val_dataset = CustomDataset(val_imgs_fold, val_masks_fold, ref_image_path, normalize=normalize_switch,cached_data=pin_memory, n_channels=n_input_channels,scale=scale_factor)
            val_loader = DataLoader(val_dataset, batch_size=1*n_devices, shuffle=False, pin_memory=pin_memory, num_workers=n_workers)

            # Start the training
            try: train_pytorch_model()

            # When the training is interrupted (Ctl + C)
            # Make sure to save a backup version and clean exit
            except KeyboardInterrupt:
                # Save the current model parameters
                if n_devices > 1:
                    torch.save(model.module.state_dict(), os.path.join(path_to_ckpts, 'backup_interruption.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(path_to_ckpts, 'backup_interruption.pth'))

                # Log the incedent
                logging.info('[ERROR] Training interrupted! parameters saved ... ')
                
                # Clean exit without any errors 
                try: sys.exit(0)
                except SystemExit: os._exit(0)
            
            # Emptying the loaders
            train_dataset.delete_cached_dataset()
            val_dataset.delete_cached_dataset()
            train_loader = []
            val_loader   = []

            # Log the incident
            logging.info(f'[INFO] Fold {i+1} : Testing the best model parameters ... ')

            # Preparing the validation dataloader
            test_dataset = CustomDataset(test_imgs_fold, test_masks_fold, ref_image_path, normalize=normalize_switch,cached_data=pin_memory, n_channels=n_input_channels, scale=scale_factor)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=pin_memory, num_workers=n_workers)

            # defining the U-Net model
            model = AttU_Net(img_ch=n_input_channels, output_ch=n_output_channels)

            # Chose the GPU cuda devices to make the training go much faster vs CPU use
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Putting the model inside the device
            model.to(device=device, dtype=torch.float32)

            model.load_state_dict(torch.load(os.path.join(path_to_ckpts, 'best_model.pth'), map_location=device))

            # Evaluation of the model
            test_loss, total_dice_coeff, total_TP, total_FP, total_TN, total_FN, total_pres, total_rec, total_acc, total_f = evaluation_pytorch_model(model, test_loader, device)

            # Getting the mean loss over all evaluation images
            test_loss = test_loss / len(test_loader)
            logging.info(f'''
                    -- Testing the best model --
                    Test loss           :  {test_loss}
                    Dice                :  {total_dice_coeff}

                    TP                  :  {total_TP}
                    FP                  :  {total_FP}
                    TN                  :  {total_TN}
                    FN                  :  {total_FN}

                    Precision           :  {total_pres}
                    Recall              :  {total_rec}
                    Accuracy            :  {total_acc}
                    F1-score            :  {total_f}

            ''')

            # Emptying the loader
            test_dataset.delete_cached_dataset()
            test_loader   = []

            # defining the U-Net model
            model = 0
            model = AttU_Net(img_ch=n_input_channels, output_ch=n_output_channels)

            # Putting the model inside the device
            model.to(device=device, dtype=torch.float32)

            # Use all the GPUs we have
            if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)
            
            # Optimizer used for the training phase
            optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.8, eps=1e-06, weight_decay=0)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)


