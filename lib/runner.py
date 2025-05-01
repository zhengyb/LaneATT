import pickle
import random
import logging
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm, trange
import shutil
import json


class Runner:
    def __init__(self, cfg, exp, device, resume=False, view=None, deterministic=False):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.logger = logging.getLogger(__name__)

        # Fix seeds
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self):
        self.exp.train_start_callback(self.cfg)
        starting_epoch = 1
        model = self.cfg.get_model()
        model = model.to(self.device)
        optimizer = self.cfg.get_optimizer(model.parameters())
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        if self.resume:
            last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(model, optimizer, scheduler)
            starting_epoch = last_epoch + 1
        max_epochs = self.cfg['epochs']
        train_loader = self.get_train_dataloader()
        loss_parameters = self.cfg.get_loss_parameters()
        for epoch in trange(starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs):
            self.exp.epoch_start_callback(epoch, max_epochs)
            model.train()
            pbar = tqdm(train_loader)
            for i, (images, labels, _) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(images, **self.cfg.get_train_parameters())
                loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Scheduler step (iteration based)
                scheduler.step()

                # Log
                postfix_dict = {key: float(value) for key, value in loss_dict_i.items()}
                postfix_dict['lr'] = optimizer.param_groups[0]["lr"]
                self.exp.iter_end_callback(epoch, max_epochs, i, len(train_loader), loss.item(), postfix_dict)
                postfix_dict['loss'] = loss.item()
                pbar.set_postfix(ordered_dict=postfix_dict)
            self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer, scheduler)

            # Validate
            if (epoch + 1) % self.cfg['val_every'] == 0:
                self.eval(epoch, on_val=True)
        self.exp.train_end_callback()

    def eval_scene(self, epoch, save_predictions=False):
        scene_list = [
            'test_curve_case',
            'test_extreme_weather_case',
            'test_night_case',
            'test_intersection_case',
            'test_up_down_case',
            'test_merge_split_case',
            'test_highway_case',
        ]
        all_metrics = {}
        for scene_name in scene_list:
            self.cfg['datasets']['test']['parameters']['split'] = scene_name
            print(f"Evaling on '{scene_name}'.........")
            metrics = self._eval(epoch, on_val=False, save_predictions=save_predictions)
            all_metrics[scene_name] = metrics
        return all_metrics

    def eval(self, epoch, on_val=False, save_predictions=False):
        print("Evaling on '%s' dataset........." % ("val" if on_val else "test"))
        metrics = self._eval(epoch, on_val=on_val, save_predictions=save_predictions)
        if not on_val:
            all_metrics = {}
            all_metrics['test'] = metrics
            metrics = self.eval_scene(epoch, save_predictions=save_predictions)
            all_metrics.update(metrics)
            print(f"All Metrics: {json.dumps(all_metrics, indent=4)}")

    def _eval(self, epoch, on_val=False, save_predictions=False):
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        self.logger.info('Loading model %s', model_path)
        model.load_state_dict(self.exp.get_epoch_model(epoch))
        model = model.to(self.device)
        model.eval()
        if on_val:
            dataloader = self.get_val_dataloader()
        else:
            dataloader = self.get_test_dataloader()
        
        dataset_name = dataloader.dataset.dataset.__class__.__name__
        split = dataloader.dataset.dataset.split
        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        if self.view:
            ret_dir = f'./datasets/{dataset_name}_{split}_ret'
            if os.path.exists(ret_dir):
                shutil.rmtree(ret_dir)
            os.makedirs(ret_dir, exist_ok=True)
        self.exp.eval_start_callback(self.cfg)
        with torch.no_grad():
            for idx, (images, _, _) in enumerate(tqdm(dataloader)):

                images = images.to(self.device)
                output = model(images, **test_parameters)
                prediction = model.decode(output, as_lanes=True)
                predictions.extend(prediction)
                if self.view:
                #if False:
                    # (B, C, H, W) -> (H, W, C)
                    # 0~1 -> 0~255
                    for i in range(len(prediction)):
                        img_idx = (idx*8)+i
                        if img_idx >= len(dataloader.dataset.annotations):
                            # if the batch size > 1, the img_idx may be out of range
                            break
                        img_path = dataloader.dataset.annotations[img_idx]['path']

                        #print(f"img_idx: {img_idx}, img_path: {img_path}")
                        img = (images[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        img, fp, fn = dataloader.dataset.draw_annotation(img_idx, img=img, pred=prediction[i])
                            #print(f"img: {idx} | fp: {fp} | fn: {fn}")
                        if self.view == 'mistakes' and fp == 0 and fn == 0:
                            continue
                        #cv2.imshow('pred', img)
                        if not isinstance(fn, list):
                            fp0 = fp
                        elif len(fp) == 0:
                            fp0 = 0
                        else:
                            assert False, f"fp is not a list: {fp}"
                            fp0 = fp[0]
                        if not isinstance(fn, list):
                            fn0 = fn
                        elif len(fn) == 0:
                            fn0 = 0
                        else:
                            fn0 = fn[0]
                        #img_name = 'image_%d_fp[%.2f]_fn{%.2f}.jpg' % ((idx*8)+i, fp0, fn0)
                        #cv2.imwrite(f'./datasets/{dataset_name}_{split}_ret/{img_name}', img)
                        old_ret_img_path = img_path.replace('.jpg', '*_pred.jpg')
                        ret_img_path = img_path.replace('.jpg', '_fp[%.2f]_fn[%.2f]_pred.jpg' % (fp0, fn0))
                        os.system("rm -f %s" % old_ret_img_path)
                        cv2.imwrite(ret_img_path, img)
                    #cv2.waitKey(0)
                #debug
                #print("break here")
                #break
        if save_predictions:
            with open('predictions.pkl', 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        metrics =  self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)
        return metrics

    def get_train_dataloader(self):
        train_dataset = self.cfg.get_dataset('train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.cfg['batch_size'],
                                                   shuffle=True,
                                                   num_workers=8,
                                                   worker_init_fn=self._worker_init_fn_)
        return train_loader

    def get_test_dataloader(self):
        test_dataset = self.cfg.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader

    def get_val_dataloader(self):
        val_dataset = self.cfg.get_dataset('val')
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.cfg['batch_size'],
                                                 shuffle=False,
                                                 num_workers=8,
                                                 worker_init_fn=self._worker_init_fn_)
        return val_loader

    def get_scene_test_dataloader(self, scene_name):
        self.cfg['datasets']['test']['parameters']['split'] = scene_name
        test_loader = self.get_test_dataloader()
        return test_loader
    
    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
