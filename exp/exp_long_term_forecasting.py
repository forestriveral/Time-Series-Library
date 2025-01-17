from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, \
    logger, config_format, train_loss_plot, speed_power_converter, \
        training_report_generator
from utils.evaluation import slide_pred_plot, slide_pred_accuracy
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import sys
import time
import json
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def logger(self, setting, action='training'):
        log_path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file_path = log_path + f'/{action}_history.txt'
        sys.stdout = logger(log_file_path)

    def recorder(self, setting, action='add', value=None):
        if action == 'init':
            self.train_recorder = {'train_loss': [], 'val_loss': [], 'test_loss': []}
        elif action == 'add':
            assert value is not None, 'Please input the value to be recorded'
            self.train_recorder['train_loss'].append(np.float(value[0]))
            self.train_recorder['val_loss'].append(np.float(value[1]))
            self.train_recorder['test_loss'].append(np.float(value[2]))
        elif action == 'save':
            assert hasattr(self, 'train_recorder'), \
                'Please initialize the recorder first by action="init"'
            record_path = os.path.join(self.args.res_path, setting)
            if not os.path.exists(record_path):
                os.makedirs(record_path)
            record_json = json.dumps(self.train_recorder, indent=4)
            with open(record_path + '/training_loss.json', 'w+') as json_file:
                json_file.write(record_json)
            train_loss_plot(self.train_recorder, record_path)
        else:
            raise ValueError('Please input the correct action')

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_start_time = time.time()
        self.recorder(setting, 'init')

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            epoch_cost_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - epoch_time))
            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_cost_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            self.recorder(setting, value=[train_loss, vali_loss, test_loss])

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.recorder(setting, 'save')

        training_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - train_start_time))
        print("Training cost time: {}".format(training_time))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = self.args.res_path + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # save the config file to results folder
        config_format(self.args, folder_path)

        test_start_time = time.time()

        self.model.eval()
        print('Testing model')
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    inputs = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = inputs.shape
                        inputs = test_data.inverse_transform(inputs.squeeze(0)).reshape(shape)
                    # gt = np.concatenate((inputs[0, :, -1], true[0, :, -1]), axis=0)
                    # pd = np.concatenate((inputs[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('\ttest shape:', preds.shape, trues.shape)

        timestamps = np.array(test_data.test_stamp)
        print('\ttime range: [', timestamps[0, 0, 0], '==>', timestamps[-1, -1, 0], ']')

        # result save
        folder_path = self.args.res_path + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        test_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - test_start_time))
        print("\tTesting cost time: {}".format(test_time))

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('\tmse:{:.2f}, mae:{:.2f}'.format(mse, mae))
        # f = open("result_long_term_forecast.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # combine the test result into one file
        test_result = np.zeros((3, preds.shape[0], preds.shape[1], preds.shape[2])).astype(np.object)
        test_result[0, :, :, :] = preds
        test_result[1, :, :, :] = trues
        test_result[2, :, :, :] = timestamps

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'time.npy', timestamps)
        np.save(folder_path + 'test.npy', test_result)

        return

    def eval(self, setting, report=False):
        folder_path = self.args.res_path + setting
        if not os.path.exists(folder_path):
            raise ValueError('No such a case folder: {}'.format(folder_path))

        # check the prediction target is speed or power
        pow_convert = speed_power_converter(self.args.target, 'power')
        spd_convert = speed_power_converter(self.args.target, 'speed')

        _ = slide_pred_plot(
            folder_path,
            convert=pow_convert,
            save=folder_path,
            )

        _ = slide_pred_plot(
            folder_path,
            convert=spd_convert,
            save=folder_path,
            )

        acc = slide_pred_accuracy(
            folder_path,
            convert=pow_convert,
            save=folder_path,
            )

        if report:
            training_report_generator(self.args, setting, acc=acc)
