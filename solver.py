import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

from resnet import *
#from visdom import Visdom
import time
import datetime

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 9
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size


class Solver(object):
    def __init__(self, config, train_loader, test_loader):
        """Initialize configurations. """
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # Model configurations.
        self.image_size = config.image_size
        
        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
#         self.max_epoch = config.max_epoch
        self.lr = config.lr
        self.num_class = config.num_class
        self.pretrain = config.pretrain
        self.num_iters_decay = config.num_iters_decay
        
        # Testing configurations.
        self.resume_iters = config.resume_iters
        
        # Directories.
        self.log_dir = config.log_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        
        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        self.train_acc_step = int(len(train_loader) * float(self.batch_size))
        
        # Miscellaneous.
        self.use_visdom = config.use_visdom
        self.model_name = config.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_acc = 0
        self.iter = 0
        
        # Build the model and visdom.
        self.build_model(config)
        
#         if self.use_visdom:
#             self.viz = Visdom()
#             self.loss_plot = self.viz.line(Y=torch.Tensor([0.]), 
#                                            X=torch.Tensor([0.]),
#                                            opts = dict(title = 'Loss for ' +
#                                                        self.model_name,
#                                                        legend=[self.model_name,],
#                                                        xlabel = 'iter',
#                                                        xtickmin = 0,
#                                                        xtickmax = 200000,
#                                                        ylabel = 'Loss',
#                                                        ytickmin = 0,
#                                                        ytickmax = 6,
#                                                    ),)
#             self.acc_plot = self.viz.line(Y=torch.Tensor([0.]),
#                                           X=torch.Tensor([0.]),
#                                           opts = dict(title = 'Test accurcay for '
#                                                       + self.model_name,
#                                                       legend=[self.model_name,],
#                                                       xlabel = 'epoch',
#                                                       xtickmin = 0,
#                                                       xtickmax = 200,
#                                                       ylabel = 'Accuracy',
#                                                       ytickmin = 0,
#                                                       ytickmax = 100,
#                                                  ),)            

    
    def build_model(self, config):
        print("    Create a model [%s]..." % self.model_name)
        if self.model_name == 'resnet50':
            self.model = resnet50(pretrained=self.pretrain, num_classes=self.num_class)

        self.model = self.model.to(self.device)
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=range(torch.cuda.device_count()))
        print("    Done")
        
        print("    Create an optimizer and a loss function")
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr, momentum=0.9, weight_decay=0.0001)
        self.criterion = nn.CrossEntropyLoss()
        print("    Done")
        
        self.print_network(self.model, self.model_name)

        
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        
            
    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr  
            
    
    def restore_model(self):
        print('Loading the trained models from step {}...'.format(self.resume_iters))
        ckpt_path = os.path.join(self.model_save_dir, '{}.ckpt'.format(self.resume_iters))
        self.model.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
        
    
    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0,1)
    
    def save_batch_results(self, images, labels, output_indice, ith):
        
        plt.clf()
        fig, ax = plt.subplots(4, 4)
        for i in range(4):
            for j in range(4):
                ax[i][j].imshow(np.transpose(self.denorm(images[4*i + j]).numpy(), (1,2,0)))
                ax[i][j].axis('off')
                ax[i][j].set_title('True:{}0s  Pred:{}0s'.format(labels[4*i + j].item()+1, output_indice[4*i + j].item()+1))
                ax[i][j].axis('off')
        plt.savefig(os.path.join(self.result_dir, '{}'.format(ith)))
        plt.close()
        print('save {}th image...'.format(ith))
        
        
    def train(self):
        print('train_acc_step: {}'.format(self.train_acc_step))
    
        # Learning reate cache for decaying.
        lr = self.lr
        
        # Start training from scratch
        start_iters = 0
        #resume code
        
        # Start training
        print('Start training...')
        start_time = time.time()
        data_iter = iter(self.train_loader)
        
        correct = 0
        total = 0
        
        for i in range(start_iters, self.num_iters):
            try:
                image, label = next(data_iter)
            except:
                data_iter = iter(self.train_loader)
                image, label = next(data_iter)
            image = image.to(self.device)
            label = label.long().to(self.device)
            
            predict = self.model(image)
            loss = self.criterion(predict, label)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # For train acc.
            _, output_index = torch.max(predict, 1)
            total += label.size(0)
            correct += (output_index == label).sum().float()
            
            # Logging.
            loss_log = {}
            loss_log['loss'] = loss.item()
            
            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss_log.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
                
#                 self.viz.line(Y=torch.Tensor([value]), 
#                          X=torch.Tensor([(i + 1)]), 
#                          win=self.loss_plot, 
#                          update='append',
#                         )
            
            # Print the predicted labels and fixed images for debugging.
            # later
#             if (i+1) % self.sample_step == 0:
#                 self.iter = i
#                 valid_acc = self.valid()
                
            if (i+1) % self.train_acc_step == 0:
                train_acc = 100 * correct / total
                print('Train accuracy: {}'.format(train_acc.item()))
                correct = 0
                total = 0

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                ckpt_path = os.path.join(self.model_save_dir, '{}.ckpt'.format(i+1))
                torch.save(self.model.state_dict(), ckpt_path)
                self.resume_iters = i+1
                print('    Save model checkpoints into {}...'.format(self.model_save_dir)) 
                valid_acc = self.valid()

            
            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0:
                self.lr = (self.lr / float(self.num_iters_decay))
                self.update_lr(self.lr)
                print ('    Decayed learning rates, lr: {}'.format(self.lr))

            
    def valid(self):
        
        self.restore_model()
        self.model.eval()
                  
        correct = 0
        total = 0
        print('length of test_loader: {}'.format(len(self.test_loader)))

        with torch.no_grad():
            for i, (image, label) in enumerate(self.test_loader):
                image = image.to(self.device)
                label = label.to(self.device)

                output = self.model(image)
                _, output_index = torch.max(output, 1) # (batch_size, 100) -> [value, index]

                total += label.size(0)
                correct += (output_index == label).sum().float()
                self.save_batch_results(image.cpu(), label.cpu(), output_index.cpu(), i)

        valid_acc = 100 * correct / total
        if valid_acc.item() > self.best_acc:
            self.best_acc = valid_acc
        print('Valid accuracy: {}, Best accuracy: {}'.format(valid_acc.item(), self.best_acc))
#         self.viz.line(Y=torch.Tensor([valid_acc.item()]), 
#                  X=torch.Tensor([(self.resume_iters)]), 
#                  win=self.acc_plot, 
#                  update='append',
#                 )
    
    
#     def test(self):
        
#         self.restore_model()
          
#         correct = 0
#         total = 0

#         for i, (image, label) in enumerate(self.test_loader):
#             image = image.to(self.device)
#             label = label.to(self.device)

#             output = self.model(image)
#             _, output_index = torch.max(output, 1) # (batch_size, 100) -> [value, index]

#             total += label.size(0)
#             age = 0
#             for i in range(16):
#                 age += output_index[i]
#             correct += (output_index == label).sum().float()
#         age_acc = (age + 0.0) / total
#         valid_acc = 100 * correct / total
# #         print('Test accuracy: {}'.format(valid_acc.item()))
#         print('average age: {}'.format(age_acc.item()))


