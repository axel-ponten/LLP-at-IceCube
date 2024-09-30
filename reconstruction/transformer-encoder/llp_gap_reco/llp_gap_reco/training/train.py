import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, model, pdf=None, device="cuda"):
        self.device = device
        
        # encoder
        self.model = model
        self.model.to(self.device)
        self.model.train()
        
        # flow?
        self.pdf = pdf
        if pdf is None:
            self.is_flow = False
            self.criterion_no_cnf = nn.MSELoss()
        else:
            self.is_flow = True
            self.pdf.to(self.device)
            self.pdf.train()
    
    def train(self,
              train_loader,
              test_loader,
              n_epochs,
              optimizer,
              scheduler,
              folder,
              start_epoch=0,
              verbose=True,
              save_freq=5,
              grad_clip = None,):
        """ Main training loop """
        # set members
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = train_loader.batch_size
        self.verbose = verbose
        self.save_freq = save_freq
        self.folder = folder
        if self.folder[-1] != "/":
            self.folder += "/"
        self.grad_clip = grad_clip # gradient clipping
        
        # register backwards hook (for gradient clipping)
        for p in self.model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip, self.grad_clip))
        if self.is_flow:
            for p in self.pdf.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -self.grad_clip, self.grad_clip))
        
        # early stopping
        self.early_stopper = EarlyStopper(patience=15, percent_tolerance=0.0)
        
        # prep some variables
        self.train_size = len(train_loader.dataset)
        self.test_size = len(test_loader.dataset)
        self.n_batches = self.train_size//self.batch_size
        self.final_epoch = start_epoch + n_epochs
        # if losses exist, open it and create lists
        if os.path.exists(self.folder + "loss.csv"):
            df = pd.read_csv(self.folder + "loss.csv")
            # trim to start epoch
            df = df[df.index < start_epoch]
            print("Opening loss file from previous training starting at epoch", start_epoch)
            self.train_loss_vals = df["train_loss"].tolist()
            self.test_loss_vals = df["test_loss"].tolist()
            if "learning_rate" in df.columns:
                self.learning_rates = df["learning_rate"].tolist()
        else:
            self.train_loss_vals = []
            self.test_loss_vals = []
            self.learning_rates = []
        
        ###### TRAIN ######
        print("Starting training loop with {} epochs".format(n_epochs))
        print("#####################################")
        self.continue_training = True
        for epoch in range(start_epoch, self.final_epoch):
            self.epoch = epoch
            # shuffle data every epoch
            self.train_loader.dataset.shuffle()
            self.model.train()
            self.train_loss = 0
            self.start_time = time.time() # epoch timer
            # iterate batches
            for i, (batch_input, batch_lens, batch_label) in enumerate(self.train_loader):
                if verbose and i%1000 == 0:
                    print(f"Batch {i}/{self.n_batches} of epoch {epoch + 1}/{self.final_epoch}")
                # process batch
                self._process_batch(batch_input, batch_lens, batch_label)
            ####################
            ### end of epoch ###
            self._end_of_epoch()
            if not self.continue_training:
                print("Training stopped at epoch", self.epoch)
                break
        #### end of training ####
        self._finish()
    
    def _process_batch(self, batch_input, batch_lens, batch_label):
        """ Process a batch. """
        # reset gradients
        self.optimizer.zero_grad()
        # compute loss
        loss = self._compute_loss(batch_input, batch_lens, batch_label)
        # compute gradient
        loss.backward()
        # clip gradients
        # if self.grad_clip is not None:
        #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        #     if self.is_flow:
        #         nn.utils.clip_grad_norm_(self.pdf.parameters(), self.grad_clip)
        
        # check gradients, only update if no NaN
        # @TODO: remove when unecessary
        try:
            self._check_gradients()
            # update train loss
            self.train_loss += loss.item()
            # update weights
            self.optimizer.step()
        except:
            print("Skipping batch due to NaN gradients")
            pass

    def _compute_loss(self, batch_input, batch_lens, batch_label):
        """ Forward pass and compute loss.
            Use the correct method depending on the model.
            
            return loss
        """
        # propagate input
        nn_output = self.model(batch_input, batch_lens)
        
        if self.is_flow:
            nn_output = nn_output.double() # make double
            batch_label = batch_label.double()
            # compute loss
            log_prob_target, log_prob_base, position_base=self.pdf(batch_label, conditional_input=nn_output)
            loss = -log_prob_target.mean()
        else:
            # compute loss
            loss = self.criterion_no_cnf(nn_output, batch_label)

        return loss
    
    def _end_of_epoch(self):
        # train loss for the epoch
        self.train_loss = self.train_loss*self.batch_size/self.train_size  # Calculate the average train loss
        self.train_loss_vals.append(self.train_loss)
        
        # test loss for the epoch
        test_loss = 0.0
        self.model.eval()  # Set the model to evaluation mode
        if self.verbose:
            with torch.no_grad():
                for batch_input, batch_lens, batch_label in self.test_loader:
                    loss = self._compute_loss(batch_input, batch_lens, batch_label)
                    test_loss += loss.item()
        
        test_loss = test_loss*self.batch_size/self.test_size  # Calculate the average test loss
        self.test_loss_vals.append(test_loss)
        
        # update learning rate?
        self.scheduler.step(test_loss)
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        print("Learning rate:", self.learning_rate)
        self.learning_rates.append(self.learning_rate)
        
        ##### aux end of epoch #####
        # print example outputs
        if self.verbose:
            print("Last three test labels:", batch_label[0:3])
            nn_output = self.model(batch_input, batch_lens)
            if self.is_flow:
                nn_output = nn_output.double()  
                y_pred = self.pdf.marginal_moments(nn_output[0:3],samplesize=300)
                ymean = y_pred["mean_0"]
                ystd  = [self._variance_from_covariance(y_pred["varlike_0"][i]) for i in range(len(y_pred["mean_0"]))] 
                print("Last three test outputs (mean and std):")
                print(ymean)
                print(ystd)
            else:
                print("Last three predictions no cnf:")
                print(nn_output[0:3])
        
        end_time = time.time()  # Stop the timer for the epoch
        epoch_time = end_time - self.start_time  # Calculate the time taken for the epoch
        print("Epoch", self.epoch + 1, "in", epoch_time, "s. Train loss", self.train_loss, ": Test loss", test_loss)
        print("#####################################")
        
        # save losses to csv
        self._save_losses()

        ####### PLOT LOSS #######
        self._plot_losses()

        # save model every 5 epochs
        if self.epoch % self.save_freq == 0:
            self._save_model()

        # save lowest loss model as model_best.pth
        if test_loss == min(self.test_loss_vals):
            torch.save(self.model.state_dict(), self.folder + "model_best.pth")
            if self.is_flow:
                torch.save(self.pdf.state_dict(), self.folder + "flow_best.pth")
                
        # early stopping
        if self.early_stopper.early_stop(test_loss):
            print("Early stopping at epoch", self.epoch)
            self._finish()

    def _save_model(self):
        model_path = self.folder + "model_epoch_{}.pth".format(self.epoch)
        torch.save(self.model.state_dict(), model_path)
        if self.is_flow:
            flow_path  = self.folder + "flow_epoch_{}.pth".format(self.epoch)
            torch.save(self.pdf.state_dict(), flow_path)

    def _plot_losses(self):
        plt.figure()
        plt.plot(self.train_loss_vals, label="train")
        plt.plot(self.test_loss_vals, label="test")
        plt.legend()
        plt.title("loss: batch size {} lr {}".format(self.batch_size, self.learning_rate))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.savefig(self.folder + "loss.png")
        plt.close()
    
    def _save_losses(self):
        df = pd.DataFrame({"train_loss": self.train_loss_vals, "test_loss": self.test_loss_vals, "learning_rate": self.learning_rates})
        df.to_csv(self.folder + "loss.csv")
    
    ## helper funcs
    def _variance_from_covariance(self, cov_mx):
        return np.sqrt(np.diag(cov_mx))
    
    def _check_gradients(self):
        # check gradients for NaN
        grads = []
        # transformer
        for param in self.model.parameters():
            if param.grad == None:
                continue
            grads.append(param.grad.view(-1))
        # flow
        if self.is_flow:
            for param in self.pdf.parameters():
                if param.grad == None:
                    continue
                grads.append(param.grad.view(-1))
        grads = torch.cat(grads)
        if torch.isnan(grads).any():

            torch.set_printoptions(threshold=10_000)
            print('########################')
            print('nan occured here')
            print('########################')
            
            # model
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    print(name)
                    print(param)
                if torch.isnan(param.grad).any():
                    print(name)
                    print(param.grad)
            # flow
            if self.is_flow:
                for name, param in self.pdf.named_parameters():
                    if torch.isnan(param).any():
                        print(name)
                        print(param)
                    if torch.isnan(param.grad).any():
                        print(name)
                        print(param.grad)
        # if any nan, skip batch
        assert not torch.isnan(grads).any(), "NaN gradients"
    
    def _finish(self):
        # save final models and losses
        model_path = self.folder + "model_final.pth"
        torch.save(self.model.state_dict(), model_path)
        if self.is_flow:
            flow_path  = self.folder + "flow_final.pth"
            torch.save(self.pdf.state_dict(), flow_path)
        self.continue_training = False

class EarlyStopper:
    """
    Class for implementing early stopping during training.

    Attributes:
        patience (int): The number of epochs to wait before stopping if the validation loss does not improve.
        percent_tolerance (float): The percentage tolerance of the running average loss for determining if the validation loss has improved.

    Methods:
        early_stop(validation_loss): Checks if the validation loss has stopped improving and returns True if early stopping criteria is met, False otherwise.
    """

    def __init__(self, patience=15, percent_tolerance=0.0):
        self.patience = patience
        self.percent_tolerance = percent_tolerance # of running avg loss

        self.counter = 0
        self.min_validation_loss = float('inf')
        # for running average of validation loss
        self.running_average = []
        self.average_length = 10

    def early_stop(self, validation_loss):
        """
        Checks if the validation loss has stopped improving and returns True if early stopping criteria is met, False otherwise.

        Args:
            validation_loss (float): The current validation loss.

        Returns:
            bool: True if early stopping criteria is met, False otherwise.
        """
        if self.percent_tolerance > 0.0:
            # running average of the last few validation losses
            if len(self.running_average) >= self.average_length:
                self.running_average.pop(0)
            self.running_average.append(validation_loss)
            tolerance = self.percent_tolerance * np.mean(self.running_average)
        else:
            tolerance = 0.0
        
        # new minimum?
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        # counter increment?
        elif validation_loss > (self.min_validation_loss + tolerance):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False