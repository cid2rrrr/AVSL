import logging
import os, pickle
from tqdm import tqdm, trange
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
import numpy as np
from tqdm.auto import tqdm

from MODULES.AVSL_model import AVSLModel
from MODULES import utils
from MODULES.utils import EvaluatorFull, AverageMeter

from DATALOADER import VideoDataLoader

from get_path import get_file_paths
from Evaluation_DATALOADER import EvaluationDataLoader
from MUSIC_DATALOADER import MUSICDataLoader
from mir_eval.separation import bss_eval_sources
from detectron2.utils.visualizer import ColorMode, Visualizer, GenericMask, VisImage
import matplotlib.colors as mplc
from detectron2.utils.colormap import random_color

SAVE_DIR = './pickle/0305test'
"""
add logging
"""
class Trainer:
    def __init__(self, cfg):

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.model = AVSLModel(cfg, training=True)
        self.model.zero_grad()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),    
            lr=0.003               
        )

        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,             
            step_size=100,         
            gamma=0.9              
        )
        
        folder_path = cfg.SOLVER.DATA_FOLDER
        self.train_dataloader = VideoDataLoader(cfg, folder_path, cfg.SOLVER.IMS_PER_BATCH)

        self.history = []
        self.num_tokens = cfg.MODEL.MASK_FORMER.NUM_OBJECT_TOKENS
        #######################


    def train(self, epochs):
        self.model.train()
        total_step = len(self.train_dataloader)
        
        for epoch in range(epochs):
            try:

                for i, batch in enumerate(self.train_dataloader):
                
                    loss = self.model(batch, mode='train')
                    losses = torch.tensor(0)
                    for key in loss.keys():
                        losses = losses + loss[key]

                    # Backward and optimize
                    self.optimizer.zero_grad()
                    losses = losses.to(self.device)
                    losses.backward() 
                    self.optimizer.step()
                    # if i%100 == 0:
                    if i%1000 == 0:
                        self.history.append(loss)

                        file_name = 'epoch_' + str(epoch) + '_batch_' + str(i) 
                        pickle_name = file_name + '.pickle'
                        pickle_path = os.path.join(SAVE_DIR, pickle_name)
                        with open (pickle_path, 'wb') as f:
                            pickle.dump(loss, f)
                        
                        weight_name = file_name + '.pth'
                        weight_path = os.path.join(SAVE_DIR, weight_name)
                        torch.save(self.model.state_dict(), weight_path)

                        print('Epoch [{}/{}], Step [{}/{}], Loss: {}'
                            .format(epoch + 1, epochs, i + 1, total_step, loss))
            except:
                print("video error")
    
    def urmp_evaluate(self, cfg, vizualization=False):
        self.model.eval()

        # URMP dataset load
        eval_dataloader = EvaluationDataLoader(cfg, cfg.TEST.DATA_FOLDER, cfg.TEST.IMS_PER_BATCH)

        # for separation
        sdr_mix_meter = AverageMeter()
        sdr_meter = AverageMeter()
        sir_meter = AverageMeter()
        sar_meter = AverageMeter()
    
        for i, batch in tqdm(enumerate(eval_dataloader)):
            if len(batch["separated_audio_paths"]) > self.num_tokens:
                continue
            mixed_audio, sep_audio_gts, outputs, _ = self.model(batch, mode='eval')
            sdr_mix, sdr, sir, sar = calc_sep_metrics(mixed_audio, sep_audio_gts, outputs)
            
        sdr_mix_meter.update(sdr_mix)
        sdr_meter.update(sdr)
        sir_meter.update(sir)
        sar_meter.update(sar)

        print('[Eval Summary]'
        'SDR_mixture: {:.4f}, SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}'
        .format(sdr_mix_meter.average(),
                sdr_meter.average(),
                sir_meter.average(),
                sar_meter.average()))
        
        if vizualization == True:
            _, _, _, processed_results = self.model(batch, mode='eval', visualization=True)
            visualized_output = visualize_mask(batch["image"][0], processed_results)  
            return visualized_output
        

    def music_evaluate(self, cfg, viz_idx_list=None):
        self.model.eval()

        # MUSIC dataset load
        eval_dataloader = MUSICDataLoader("DATA/MUSIC_DUET/frames/", "DATA/MUSIC_DUET/audios",
                                          "DATA/metadata/music_duet.json", cfg.TEST.IMS_PER_BATCH)
        
        # for localization
        evaluator_0 = EvaluatorFull()
        evaluator_1 = EvaluatorFull()
        best_precision, best_ap, best_f1 = 0., 0., 0.
        viz_list = []

        for i, batch in tqdm(enumerate(eval_dataloader)):
            if batch["bboxes"]["gt_map"].shape[1] > self.num_tokens:
                continue
            if viz_idx_list is None or i not in viz_idx_list:
                _, _, outputs, _ = self.model(batch, mode='eval')
            # elif :
                # _, _, outputs, _ = self.model(batch, mode='eval')
            else:
                _, _, outputs, processed_results = self.model(batch, mode='eval', visualization=True)
                visualized_output = visualize_mask(batch["image"][0], processed_results)
                viz_list.append(visualized_output)
                return viz_list

            calc_loc_metrics(batch["bboxes"], outputs, evaluator_0, evaluator_1)
            
        precision = (evaluator_0.precision_at_10() + evaluator_1.precision_at_10()) / 2
        ap = (evaluator_0.piap_average() + evaluator_1.piap_average()) / 2
        f1 = (evaluator_0.f1_at_30() + evaluator_1.f1_at_30()) / 2
        if precision > best_precision:
            best_precision, best_ap, best_f1 = precision, ap, f1

        print('[Eval Summary]'
        'Precision: {:.4f}, AP: {:.4f}, F1: {:.4f}'.format(precision, ap, f1))
        print('best Precision: {:.4f}, best AP: {:.4f}, best F1: {:.4f}'.format(best_precision, best_ap, best_f1))
        
        # if vizualization == True:
        #     _, _, _, processed_results = self.model(batch, mode='eval', visualization=True)
        #     visualized_output = visualize_mask(batch["image"][0], processed_results)  
        #     return visualized_output


def calc_sep_metrics(mixed_audio, sep_audio_gts, outputs):
    # meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    B = mixed_audio.shape[0] #1

    # loop over each sample
    for b in range(B):
        N = sep_audio_gts.shape[1]

        gts_wav = [None for n in range(N)]
        preds_wav = [t.data.cpu().numpy() for t in outputs["sep_audio_wavs_wo_noise"][b][:N]]
        valid=True
        for n in range(N):
            gts_wav[n] = sep_audio_gts[b][n].data.cpu().numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        sdr, sir, sar, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray(preds_wav),
                False
                )
                # gts_wav.data.cpu().numpy(),
                # preds_wav.data.cpu().numpy(),
        sdr_mix, _, _, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mixed_audio[0][:,0].data.cpu().numpy() for n in range(N)]),
                False)
    # return np.asarray(gts_wav), np.asarray(preds_wav)
    # return sdr_mix, sdr, sir, sar
        sdr_mix_meter.update(sdr_mix.mean())
        sdr_meter.update(sdr.mean())
        sir_meter.update(sir.mean())
        sar_meter.update(sar.mean())
    
    return [sdr_mix_meter.average(),
            sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average()]

def calc_loc_metrics(bboxes, outputs, evaluator_0, evaluator_1):
    """
    bboxes: batch["bboxes"]
    outputs: return of model(eval)
    """
    # meters
    # evaluator_0 = EvaluatorFull()
    # evaluator_1 = EvaluatorFull()
    
    mask_pred_results = outputs["mask_pred_results"].data.cpu().numpy()

    tau = 0.03
    av_min, av_max = -1. / tau, 1. / tau
    min_max_norm = lambda x, xmin, xmax: (x - xmin) / (xmax - xmin)

    B = outputs["pred_masks"].shape[0] # 4
    for b in range(B):
        gt_map = bboxes['gt_map'][b].data.cpu().numpy()     # (2, 224, 224)
        bb = bboxes['bboxes'][b]
        bb = bb[bb[:, 0] >= 0].numpy().tolist()

        N = len(bb)
        for n in range(N):
            hw = mask_pred_results[b, n, 0].size
            scores = min_max_norm(mask_pred_results[b, n, 0], av_min, av_max)
            pred = utils.normalize_img(scores)
            conf = np.sort(scores.flatten())[-hw//4:].mean()
            thr = np.sort(pred.flatten())[int(hw*0.5)]

            if n == 0:
                evaluator_0.update(bb, gt_map[n], conf, pred, thr, None)
            elif n == 1:
                evaluator_1.update(bb, gt_map[n], conf, pred, thr, None)


def visualize_mask(input_image, processed_results):
    """
    input_image: batched_inputs["image"][0]
    processed_results: list(dict{"seg": torch.Size[4, 640, 640]})
    """
    # visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
    img_rgb = input_image.permute(1,2,0).data.cpu().numpy()[:,:,::-1]
    img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
    visualized_output = VisImage(img, scale=1.0)

    # def draw_sem_seg(self, sem_seg, area_threshold=None, alpha=0.8)
    # visualizer.draw_sem_seg(predictions["sem_seg"].argmax(dim=0).to(self.cpu_device))
    sem_seg = processed_results[0]["seg"].argmax(dim=0).to("cpu") # 4개의 prediction 중 위치 별로 가장 높은 score가 있는 mask의 idx

    if isinstance(sem_seg, torch.Tensor):
        sem_seg = sem_seg.data.cpu().numpy()
    labels, areas = np.unique(sem_seg, return_counts=True)
    sorted_idxs = np.argsort(-areas).tolist()
    labels = labels[sorted_idxs]

    # for label in filter(lambda l: l < len(self.metadata.stuff_classes), labels):
    alpha=0.8
    for label in labels:

        binary_mask = (sem_seg == label).astype(np.uint8)
        
        color = random_color(rgb=True, maximum=1)
        color = mplc.to_rgb(color)
        has_valid_segment = False
        binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
        mask = GenericMask(binary_mask, visualized_output.height, visualized_output.width)
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])

        rgba = np.zeros(shape2d + (4,), dtype="float32")
        rgba[:, :, :3] = color
        rgba[:, :, 3] = (mask.mask == 1).astype("float32") * alpha
        has_valid_segment = True
        visualized_output.ax.imshow(rgba, extent=(0, visualized_output.width, visualized_output.height, 0))
    
    return visualized_output



from MODULES.AVSL_model_light import AVSLModelLight
from MODULES.criterion_custom import AS_loss
from DATALOADER import ImageDataLoader
import traceback

class LightTrainer:
    def __init__(self, cfg):

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.model = AVSLModelLight(cfg, training=True)
        self.model.zero_grad()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),    
            lr=0.003               
        )

        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,             
            step_size=100,         
            gamma=0.9              
        )
        
        folder_path = cfg.SOLVER.DATA_FOLDER
        self.train_dataloader = VideoDataLoader(cfg, folder_path, cfg.SOLVER.IMS_PER_BATCH)
        

        img_folder_path = 'DATA/images'
        aud_folder_path = 'DATA/audios'
        self.image_dataloader = ImageDataLoader(cfg, img_folder_path, aud_folder_path, 4)

        self.history = []
        self.num_tokens = cfg.MODEL.MASK_FORMER.NUM_OBJECT_TOKENS
        #######################


    def train(self, epochs):
        self.model.train()
        total_step = len(self.train_dataloader)
        
        try:
            for epoch in range(epochs):
                
                # for i, batch in enumerate(self.train_dataloader):
                for i, batch in enumerate(self.image_dataloader):

                    outputs = self.model(batch, mode='train')
                    as_loss = AS_loss(outputs["mixed_audio_spec"], outputs["sep_audio_specs"])
                    
                    # Backward and optimize
                    self.optimizer.zero_grad()
                    as_loss = as_loss.to(self.device)
                    as_loss.backward() 
                    self.optimizer.step()
                    # if i%100 == 0:
                    if i%1000 == 0:
                        self.history.append(as_loss)

                        file_name = 'epoch_' + str(epoch) + '_batch_' + str(i) 
                        pickle_name = file_name + '.pickle'
                        pickle_path = os.path.join(SAVE_DIR, pickle_name)
                        with open (pickle_path, 'wb') as f:
                            pickle.dump(as_loss, f)
                        
                        weight_name = file_name + '.pth'
                        weight_path = os.path.join(SAVE_DIR, weight_name)
                        torch.save(self.model.state_dict(), weight_path)

                        print('Epoch [{}/{}], Step [{}/{}], Loss: {}'
                        .format(epoch + 1, epochs, i + 1, total_step, as_loss))

        except Exception as e:
                err = traceback.format_exc()
                save_string_to_file(err, "pickle/0305test/err.txt")
                print(err)


def save_string_to_file(string, filename):
    with open(filename, 'w') as f:
        f.write(string)



class URMPTrainer:
    def __init__(self, cfg):

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.model = AVSLModelLight(cfg, training=True)
        self.model.zero_grad()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),    
            lr=0.003               
        )

        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,             
            step_size=100,         
            gamma=0.9              
        )
        
        folder_path = cfg.SOLVER.DATA_FOLDER
        self.train_dataloader = EvaluationDataLoader(cfg, cfg.TEST.DATA_FOLDER, cfg.TEST.IMS_PER_BATCH)

        self.history = []
        self.num_tokens = cfg.MODEL.MASK_FORMER.NUM_OBJECT_TOKENS
        #######################


    def train(self, epochs):
        self.model.train()
        total_step = len(self.train_dataloader)
        
        try:
            for epoch in range(epochs):
                
                for i, batch in enumerate(self.train_dataloader):

                    outputs = self.model(batch, mode='train')
                    as_loss = AS_loss(outputs["mixed_audio_spec"], outputs["sep_audio_specs"])
                    
                    # Backward and optimize
                    self.optimizer.zero_grad()
                    as_loss = as_loss.to(self.device)
                    as_loss.backward() 
                    self.optimizer.step()
                    # if i%100 == 0:
                    if i%43 == 0:
                        self.history.append(as_loss)

                        file_name = 'epoch_' + str(epoch) + '_batch_' + str(i) 
                        pickle_name = file_name + '.pickle'
                        pickle_path = os.path.join("./pickle/0305test_urmp", pickle_name)
                        with open (pickle_path, 'wb') as f:
                            pickle.dump(as_loss, f)
                        
                        weight_name = file_name + '.pth'
                        weight_path = os.path.join("./pickle/0305test_urmp", weight_name)
                        torch.save(self.model.state_dict(), weight_path)

                        print('Epoch [{}/{}], Step [{}/{}], Loss: {}'
                        .format(epoch + 1, epochs, i + 1, total_step, as_loss))

        except Exception as e:
                err = traceback.format_exc()
                # save_string_to_file(err, "pickle/0305test/err.txt")
                print(err)
