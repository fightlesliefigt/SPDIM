import torch
import geoopt
from sklearn.metrics import balanced_accuracy_score
from .callbacks import Callback
from geoopt.optim import RiemannianAdam
from spdnets.manifolds import SymmetricPositiveDefinite 
from . import functionals as fn
from torch import optim

class Trainer:

    def __init__(self, max_epochs, callbacks, min_epochs=None, loss=None, device=None, dtype=None):

        self.min_epochs = min_epochs
        self.epochs = max_epochs
        self.loss_fn = loss
        self.current_epoch = 0
        self.current_step = 0
        self.records = []
        for callback in callbacks:
            assert(isinstance(callback, Callback))
        self.callbacks = callbacks

        self.device_ = device
        self.dtype_ = dtype

        self.stop_fit_ = False
        self.optimizer = None

    def fit(self, model : torch.nn.Module, train_dataloader : torch.utils.data.DataLoader, val_dataloader : torch.utils.data.DataLoader,parameter_t,fm_mean=None):

        model = model.to(dtype=self.dtype_, device=self.device_)

        self.optimizer = model.configure_optimizers()

        [callback.on_fit_start(self, model) for callback in self.callbacks]

        for epoch in range(self.epochs):

            self.current_epoch = epoch
            [callback.on_train_epoch_start(self, model) for callback in self.callbacks]

            self.train_epoch(model, train_dataloader,parameter_t,fm_mean)

            trn_res = self.test(model, train_dataloader,parameter_t,fm_mean)
            trn_res = {f'trn_{k}': v for k, v in trn_res.items()}

            val_res = self.test(model, val_dataloader,parameter_t,fm_mean)
            val_res = {f'val_{k}': v for k, v in val_res.items()}

            self.log_dict(trn_res)
            self.log_dict(val_res)

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                log_dict = trn_res | val_res
                print(f'epoch={epoch:3d} gd-step={self.current_step:5d}', end=' ')
                [print(f"{k + '=':10}{v:6.4f}", end=' ') for k,v in log_dict.items()]
                print('')                                                                 #换行


            [callback.on_train_epoch_end(self, model) for callback in self.callbacks]

            if self.stop_fit_:
                break

        [callback.on_fit_end(self, model) for callback in self.callbacks]

    def stop_fit(self):
        if self.min_epochs and self.current_epoch > self.min_epochs:
            self.stop_fit_ = True
        elif self.min_epochs is None:
            self.stop_fit_ = True
        

    def train_epoch(self, model : torch.nn.Module, train_dataloader : torch.utils.data.DataLoader,parameter_t,fm_mean=None):

        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            [callback.on_train_batch_start(self, model, batch, batch_idx) for callback in self.callbacks]
            features, y = batch
            features['inputs'] = features['inputs'].to(dtype=self.dtype_, device=self.device_)
            y = y.to(device=self.device_)
            pred = model(**features,parameter_t=parameter_t,fm_mean=fm_mean)
            loss = self.loss_fn(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.current_step += 1
 

    def predict(self, model : torch.nn.Module, dataloader : torch.utils.data.DataLoader,parameter_t,fm_mean=None):

        model.eval()
        y_hat = []
        for _, (features, y) in enumerate(dataloader):
            features['inputs'] = features['inputs'].to(dtype=self.dtype_, device=self.device_)
            y = y.to(device=self.device_)
            pred = model(**features,parameter_t=parameter_t,fm_mean=fm_mean)
            y_hat.append(pred.argmax(1))
        return pred,y_hat


    def test(self, model : torch.nn.Module, dataloader : torch.utils.data.DataLoader,parameter_t,fm_mean=None):

        model.eval()
        loss = 0

        y_true = []
        y_hat = []

        with torch.no_grad():
            for batch_ix, (features, y) in enumerate(dataloader):
                features['inputs'] = features['inputs'].to(dtype=self.dtype_, device=self.device_)
                y = y.to(device=self.device_)
                pred = model(**features,parameter_t=parameter_t,fm_mean=fm_mean)
                loss += self.loss_fn(pred, y).item()
                y_true.append(y)
                y_hat.append(pred.argmax(1))

        loss /= batch_ix + 1

        score = balanced_accuracy_score(torch.cat(y_true).detach().cpu().numpy(), torch.cat(y_hat).detach().cpu().numpy()).item()
        return dict(loss=loss, score=score)


    def log_dict(self, dictionary):
        self.records.append(dictionary | dict(epoch=self.current_epoch))
    
    def get_information_maximization_bias(self, model : torch.nn.Module, test_dataloader : torch.utils.data.DataLoader, parameter_t,epochs=50, lr=0.01,ts=2):
        fm_mean=fn.spd_mean_kracher_flow(model.get_spdnet_data())
        fm_mean = geoopt.ManifoldParameter(fm_mean, manifold=SymmetricPositiveDefinite())  # constraint the optimization on the spd manifold
        fm_mean.requires_grad=True
        optimizer = RiemannianAdam([fm_mean], lr=lr)
        best_loss = float('inf')
        best_mean = fm_mean.clone()      
        for epoch in range(epochs):
            optimizer.zero_grad()
            prediction,_ = self.predict(model, dataloader=test_dataloader, parameter_t=parameter_t,fm_mean=fm_mean)
            loss=fn.im_loss(prediction,ts)
            loss.backward()
            optimizer.step()                      
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_mean = fm_mean.clone()
        return  best_mean

    
    # get information maximization bias on the geodesic the geodesic bewteen the domain-specific mean and the indentiy matrix
    def get_information_maximization_geodesic(self, model : torch.nn.Module, test_dataloader : torch.utils.data.DataLoader, parameter_t,epochs=50, lr=0.01,ts=2):
        parameter_t.requires_grad=True
        optimizer = optim.Adam([parameter_t], lr=lr)
        best_loss = float('inf')
        best_t = parameter_t.clone()
        for epoch in range(epochs):
            optimizer.zero_grad()
            prediction,_ = self.predict(model, dataloader=test_dataloader, parameter_t=parameter_t)
            loss = fn.im_loss(prediction,ts)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                parameter_t.clamp_(0, 2)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_t = parameter_t.clone()
        return best_t

    
 