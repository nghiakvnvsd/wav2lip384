from os.path import dirname, join, basename, isfile
import math
from tqdm import tqdm
import traceback
import pandas as pd
import time
import datetime

from models import SyncNet_color as SyncNet
from models import Wav2Lip, Wav2Lip_disc_qual

import audio

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
# from torch.utils.tensorboard import SummaryWriter
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list
import torch.multiprocessing as mp
import torch.distributed as dist
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt

# TODO: change required to True
parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model WITH the visual quality discriminator')

# parser.add_argument("--data_root", help="Root folder of the preprocessed dataset", required=False, default="/home/vtcai/vtc_data", type=str)
parser.add_argument("--data_root", help="Root folder of the preprocessed dataset", required=False, default="", type=str)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=False, default="./weight/wav", type=str)
parser.add_argument('--log_dir', help='Write log files to this directory', required=False, default="./logs/wav", type=str)

parser.add_argument('--exp_num', help='ID number of the experiment', required=False, default="ex_wav2lip_margin", type=str)
parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator', default="syncnet.pth", required=False, type=str)

parser.add_argument('--checkpoint_path', help='Resume generator from this checkpoint', default=None, type=str)
parser.add_argument('--disc_checkpoint_path', help='Resume quality disc from this checkpoint', default=None, type=str)

args = parser.parse_args()

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
# use_cuda = False
best_loss = 10000
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16
mask_margin = 8
upper_offset = 20
use_aug = True

img_size = 192
width = img_size - mask_margin * 2
height = img_size - img_size//2 - mask_margin
mask = np.zeros((height, width))
top_mask = np.linspace(1, 0, mask_margin)
top_mask = np.repeat(np.expand_dims(top_mask, axis=-1), width, axis=-1)
bottom_mask = np.linspace(0, 1, mask_margin)
bottom_mask = np.repeat(np.expand_dims(bottom_mask, axis=-1), width, axis=-1)
left_mask = np.linspace(1, 0, mask_margin)
left_mask = np.repeat(np.expand_dims(left_mask, axis=0), height, axis=0)
right_mask = np.linspace(0, 1, mask_margin)
right_mask = np.repeat(np.expand_dims(right_mask, axis=0), height, axis=0)
mask[:mask_margin, :] = top_mask
mask[-mask_margin:, :] = bottom_mask

mask[:, :mask_margin] = np.maximum(mask[:, :mask_margin], left_mask)
mask[:, -mask_margin:] = np.maximum(mask[:, -mask_margin:], right_mask)
  
full_mask = np.ones((img_size, img_size))
full_mask[full_mask.shape[0]//2:-mask_margin, mask_margin:-mask_margin] = mask
full_mask = np.expand_dims(full_mask, axis=-1)
full_mask = np.repeat(full_mask, 3, axis=-1)
hparams.set_hparam('img_size', 384)
hparams.set_hparam('batch_size', 32)

def mask_mel(crop_mel):
    block_size = 0.1
    time_size = math.ceil(block_size * crop_mel.shape[0])
    freq_size = math.ceil(block_size * crop_mel.shape[1])
    time_lim = crop_mel.shape[0] - time_size
    freq_lim = crop_mel.shape[1] - freq_size

    time_st = random.randint(0, time_lim)
    freq_st = random.randint(0, freq_lim)

    mel = crop_mel.copy()
    mel[time_st:time_st+time_size] = -4.
    mel[:, freq_st:freq_st + freq_size] = -4.

    return mel

class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, f'{frame_id:05}.jpg')
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames, return_mask = False):
        if window_fnames is None: return None
        window = []
        if return_mask:
            masks = []
            img_masks = []
            mask = full_mask.copy()

        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None

            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                if return_mask:
                    mask_img = img * mask
            except Exception as e:
                return None

            window.append(img)
            if return_mask:
                img_masks.append(mask_img)
                masks.append(mask)

        if return_mask:
            return window, img_masks, masks
        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = (np.asarray(window) - 127.5) / 127.5
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                # print("Len", vidname)
                continue

            idx_mel = random.randint(0, len(self.all_videos) - 1)
            vidname_mel = self.all_videos[idx_mel]
            img_names_mel = list(glob(join(vidname_mel, '*.jpg')))
            if len(img_names_mel) <= 3 * syncnet_T:
                # print("Len", vidname)
                continue
            img_name_mel = random.choice(img_names_mel)

            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                # print("Wrong 1", window_fnames, wrong_window_fnames)
                continue

            window = self.read_window(window_fnames)
            if window is None:
                # print("Windoe", vidname)
                continue

            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                # print("wrong Windoe", vidname)
                continue

            try:
                mel_out_path = join(vidname_mel, "mel.npy")
                if os.path.isfile(mel_out_path):  # x50 times faster - 0.002 -> 0.01s
                    with open(mel_out_path, "rb") as f:
                        orig_mel = np.load(f)
                else:
                    wavpath = os.path.join(vidname, "synced_audio.wav")
                    if not os.path.isfile(wavpath):
                        au_names = list(glob(join(vidname, '*.wav')))
                        au_path = au_names[0]
                        status = os.system(f"ffmpeg -i {au_path} -ar 16000 {wavpath}")

                    wav = audio.load_wav(wavpath, hparams.sample_rate)

                    orig_mel = audio.melspectrogram(wav).T  # 0.2 -> 0.9s
                    with open(mel_out_path, "wb") as f:
                        np.save(f, orig_mel)
            except Exception as e:
                # print("Mel", vidname)
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name_mel)

            if (mel.shape[0] != syncnet_mel_step_size):
                # print("Mel shape", vidname)
                continue

            # mel augmentation
            if use_aug and random.random() < 0.3:
                mel = mask_mel(mel)

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name_mel)
            if indiv_mels is None:
                continue

            # ground truth images
            window = self.prepare_window(window)
            y = window.copy()

            # mask the lower half of the images
            window[:, :, window.shape[2]//2:-mask_margin, mask_margin:-mask_margin] = -1

            # reference images
            wrong_window = self.prepare_window(wrong_window)
            x = np.concatenate([window, wrong_window], axis=0)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            return x, indiv_mels, mel, y


def save_sample_images(x, g, gt, global_step, checkpoint_dir):
    x = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 127.5 + 127.5).astype(np.uint8)
    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 127.5 + 127.5).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 127.5 + 127.5).astype(np.uint8)

    refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((refs, inps, g, gt), axis=-2)
    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])


logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    # d = (d +1 ) / 2
    loss = logloss(d.unsqueeze(1), y)
    return loss

device = torch.device("cuda" if use_cuda else "cpu")

recon_loss = nn.L1Loss()
def get_sync_loss(mel, g, syncnet):
    if syncnet is None:
        return torch.Tensor([10])
    print(g.shape)
    g = g[:, :, :, g.size(3)//2:]
    print(g.shape)
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

def viz_loss(num_epochs, arr_disc_fake_loss, arr_disc_real_loss, arr_perceptual_loss, step):
    """ Visualize loss for the generator, discriminator """
    # Set style, figure size
    plt.style.use('ggplot')
    plt.rcParams["figure.figsize"] = (8,6)

    # Plot Discriminator loss in red, Generator loss in green
    plt.plot(np.linspace(1, num_epochs, len(arr_disc_fake_loss)),
                arr_disc_fake_loss,
                'r')
    plt.plot(np.linspace(1, num_epochs, len(arr_disc_real_loss)),
                arr_disc_real_loss,
                'g')

    plt.plot(np.linspace(1, num_epochs, len(arr_perceptual_loss)),
            arr_perceptual_loss,
            'y')

    # Add legend, title
    plt.legend(['arr_disc_fake_loss', 'arr_disc_real_loss', 'arr_perceptual_loss'])
    plt.title("arr_disc_fake_loss, arr_disc_real_loss and arr_perceptual_loss")
    plt.savefig(f"/home/lenghia/loss_figures/loss_{step}")
    plt.clf()


def train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None, log_interval=None,syncnet=None):
    global global_step, global_epoch
    resumed_step = global_step

    if not os.path.isdir(args.log_dir): os.makedirs(args.log_dir)
    logger = CSVLogger(args.log_dir, name=f"train{args.exp_num}")
    valLogger = CSVLogger(args.log_dir, name=f"val{args.exp_num}")
    bce_loss = nn.BCELoss()
    syncnet_wt = hparams.syncnet_wt
    arr_disc_fake_loss = []
    arr_disc_real_loss = []
    arr_perceptual_loss = []
    while global_epoch < nepochs:
        try:
            stop_training = False
            print('Starting Epoch: {}'.format(global_epoch))

            running_sync_loss, running_l1_loss, disc_loss, running_perceptual_loss = 0., 0., 0., 0.
            running_disc_real_loss, running_disc_fake_loss = 0., 0.
            st = time.time()
            offset = 0
            for step, (x, indiv_mels, mel, gt) in enumerate(train_data_loader):
                load_time = time.time() - st
                st = time.time()
                disc.train()
                model.train()

                x = x.to(device)
                # mel = mel.to(device)
                # indiv_mels = indiv_mels.to(device)
                gt = gt.to(device)

                ### Train generator now. Remove ALL grads.
                optimizer.zero_grad()
                disc_optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=False):
                    g = model(indiv_mels, x)

                    if hparams.syncnet_wt > 0. and syncnet is not None:
                        try:
                            sync_loss = get_sync_loss(mel, g, syncnet)
                        except:
                            offset += 1
                            continue
                    else:
                        sync_loss = 0.

                    if hparams.disc_wt > 0.:
                        # with torch.cuda.amp.autocast(enabled=False):
                        pred = disc(g)
                        target = torch.ones((len(pred), 1))
                        target = target.to(device)
                        # perceptual_loss = torch.mean(torch.square(pred - target))
                        perceptual_loss = F.binary_cross_entropy(pred, target)
                        # perceptual_loss = F.mse_loss(pred, target)
                    else:
                        perceptual_loss = 0.

                    with torch.cuda.amp.autocast(enabled=False):
                        ### mask 1/2 face
                        l1loss = recon_loss(g[:, :, :, :g.size(3)//2],
                                            gt[:, :, :, :gt.size(3)//2])
                        l1loss += recon_loss(g[:, :, :, g.size(3)//2:-mask_margin, :mask_margin],
                                            gt[:, :, :, gt.size(3)//2:-mask_margin, :mask_margin])
                        l1loss += recon_loss(g[:, :, :, g.size(3)//2:-mask_margin, -mask_margin:],
                                             gt[:, :, :, gt.size(3)//2:-mask_margin, -mask_margin:])
                        l1loss += recon_loss(g[:, :, :, -mask_margin:],
                                            gt[:, :, :, -mask_margin:])
                        
                        loss = syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
                                                (1. - syncnet_wt - hparams.disc_wt) * l1loss

                loss.backward()
                optimizer.step()

                ### Remove all gradients before Training disc
                disc_optimizer.zero_grad()

                pred = disc(gt)
                target = torch.ones((len(pred), 1)) * 0.9995
                target = target.to(device)
                disc_real_loss = F.binary_cross_entropy(pred, target)

                disc_real_loss.backward()

                pred = disc(g.detach())
                target = torch.zeros((len(pred), 1)) + 0.0005
                target = target.to(device)
                disc_fake_loss = F.binary_cross_entropy(pred, target)
    
                disc_fake_loss.backward()

                disc_optimizer.step()

                running_disc_real_loss += disc_real_loss.item()
                arr_disc_real_loss.append(running_disc_real_loss/(step+1-offset))
                running_disc_fake_loss += disc_fake_loss.item()
                arr_disc_fake_loss.append(running_disc_fake_loss/(step+1-offset))

                if global_step % checkpoint_interval == 0:
                    save_sample_images(x, g, gt, global_step, checkpoint_dir)

                del x, g, gt, indiv_mels, mel

                # Logs
                global_step += 1
                cur_session_steps = global_step - resumed_step

                running_l1_loss += l1loss.item()
                if hparams.syncnet_wt > 0. and syncnet is not None:
                    running_sync_loss += sync_loss.item()
                else:
                    running_sync_loss += 0.

                if hparams.disc_wt > 0.:
                    running_perceptual_loss += perceptual_loss.item()
                else:
                    running_perceptual_loss += 0.
                arr_perceptual_loss.append(running_perceptual_loss/(step+1-offset))
                # logs
                if global_step == 1 or global_step % log_interval == 0:
                    logger.log_metrics({
                        "Generator/l1_loss/train": running_l1_loss/(step+1-offset),
                        "syncnet_wt": syncnet_wt,
                        "Generator/sync_loss/train": running_sync_loss/(step+1-offset),
                        "Generator/perceptual_loss/train": running_perceptual_loss/(step+1-offset),
                        "Discriminator/fake_loss/train": running_disc_fake_loss/(step+1-offset),
                        "Discriminator/real_loss/train": running_disc_real_loss/(step+1-offset)
                    }, step=global_step)
                    logger.save()

                if global_step % checkpoint_interval == 0:
                    save_checkpoint(
                        model, optimizer, global_step, checkpoint_dir, global_epoch, prefix="gen_")
                    save_checkpoint(disc, disc_optimizer, global_step, checkpoint_dir, global_epoch, prefix='disc_')

                if global_step > 100000:
                    stop_training = True
                    break

                train_time = time.time() - st
                print('Step {} | L1: {:.6} | Sync_wt: {:.4f} Sync: {:.6}, Percep: {:.6} | Fake: {:.6}, Real: {:.6} | Load: {:.6}, Train: {:.6}'
                      .format(global_step,
                              running_l1_loss / (step + 1-offset),
                              syncnet_wt,
                              running_sync_loss / (step + 1-offset),
                              running_perceptual_loss / (step + 1-offset),
                              running_disc_fake_loss / (step + 1-offset),
                              running_disc_real_loss / (step + 1-offset),
                              load_time, train_time))
                st = time.time()

                if disc_fake_loss.item() > 90 or disc_real_loss.item() > 90:
                    stop_training = True
                    break

            if stop_training:
                print("Stop training...")
                break

            global_epoch += 1
        except Exception as e:
            print(e)
            # print("KeyboardInterrupt")
            break
    print("Saving models and logs...")
    save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch, prefix="gen_")
    save_checkpoint(disc, disc_optimizer, global_step, checkpoint_dir, global_epoch, prefix='disc_')
    logger.save()
    valLogger.save()
    # fidLogger.save()

def eval_model(test_data_loader, global_step, device, model, disc, logger,syncnet):
    eval_steps = 10
    print('Evaluating for {} steps'.format(eval_steps))
    running_sync_loss, running_l1_loss, running_disc_real_loss, running_disc_fake_loss, running_perceptual_loss = [], [], [], [], []
    offset = 0
    for step, (x, indiv_mels, mel, gt) in enumerate((test_data_loader)):

        model.eval()
        disc.eval()

        x = x.to(device)
        mel = mel.to(device)
        indiv_mels = indiv_mels.to(device)
        gt = gt.to(device)

        pred = disc(gt)

        target = torch.ones((len(pred), 1))
        target = target.to(device)
        
        disc_real_loss = F.binary_cross_entropy(pred, target)

        g = model(indiv_mels, x)
        pred = disc(g)
        target = torch.zeros((len(pred), 1))
        target = target.to(device)
        disc_fake_loss = F.binary_cross_entropy(pred, target)

        running_disc_real_loss.append(disc_real_loss.item())
        running_disc_fake_loss.append(disc_fake_loss.item())

        try:
            sync_loss = get_sync_loss(mel, g, syncnet).to(device)
        except:
            offset += 1
            continue

        if hparams.disc_wt > 0.:
            # perceptual_loss = disc.perceptual_forward(g)
            pred = disc(g)
            target = torch.ones((len(pred), 1))
            target = target.to(device)
            perceptual_loss = F.binary_cross_entropy(pred, target)
        else:
            perceptual_loss = 0.

        # l1loss = recon_loss(g * masks, gt * masks)
        l1loss = recon_loss(g[:, :, :, :g.size(3)//2],
                            gt[:, :, :, :gt.size(3)//2])
        l1loss += recon_loss(g[:, :, :, g.size(3)//2:-mask_margin, :mask_margin],
                            gt[:, :, :, gt.size(3)//2:-mask_margin, :mask_margin])
        l1loss += recon_loss(g[:, :, :, g.size(3)//2:-mask_margin, -mask_margin:],
                             gt[:, :, :, gt.size(3)//2:-mask_margin, -mask_margin:])
        l1loss += recon_loss(g[:, :, :, -mask_margin:],
                            gt[:, :, :, -mask_margin:])
        l1loss = recon_loss(g, gt)

        loss = hparams.syncnet_wt * sync_loss + hparams.disc_wt * perceptual_loss + \
                                (1. - hparams.syncnet_wt - hparams.disc_wt) * l1loss

        running_l1_loss.append(l1loss.item())
        running_sync_loss.append(sync_loss.item())

        if hparams.disc_wt > 0.:
            running_perceptual_loss.append(perceptual_loss.item())
        else:
            running_perceptual_loss.append(0.)

        if step > eval_steps: break

    print('L1: {:.6}, Sync: {:.6}, Percep: {:.6} | Fake: {:.6}, Real: {:.6}'
          .format(sum(running_l1_loss) / len(running_l1_loss),
                  sum(running_sync_loss) / len(running_sync_loss),
                  sum(running_perceptual_loss) / len(running_perceptual_loss),
                  sum(running_disc_fake_loss) / len(running_disc_fake_loss),
                  sum(running_disc_real_loss) / len(running_disc_real_loss)))
    # logs
    logger.log_metrics({
        "Generator/l1_loss/val": np.mean(running_l1_loss),
        "Generator/sync_loss/val": np.mean(running_sync_loss),
        "Generator/perceptual_loss/val": np.mean(running_perceptual_loss),
        "Discriminator/fake_loss/val": np.mean(running_disc_fake_loss),
        "Discriminator/real_loss/val": np.mean(running_disc_real_loss)
    })
    logger.save()
    return sum(running_sync_loss) / len(running_sync_loss)

def upload_file(path):
    pass

def save_ckpt(model, optimizer, step, checkpoint_dir, epoch, model_name):
    checkpoint_path = join(
        checkpoint_dir, model_name)

    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "best_loss": best_loss,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix='', loss_val=1000):
    # save best.pth
    global best_loss
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    date = str(datetime.datetime.now()).split(" ")[0]
    post_fix = f'checkpoint_{hparams.img_size}_{hparams.batch_size}_{global_step:09d}_{date}.pth'
    if loss_val <= best_loss:
        best_loss = loss_val
        best_name = f"{prefix}best_wav128_1e4.pth"
        save_ckpt(model, optimizer, step, checkpoint_dir, epoch, best_name)

    last_name = f"{prefix}last_wav128_1e4.pth"
    save_ckpt(model, optimizer, step, checkpoint_dir, epoch, last_name)

    save_ckpt(model, optimizer, step, checkpoint_dir, epoch, f"{prefix}{post_fix}")

    ckpt_list = os.listdir(checkpoint_dir)
    ckpt_list = [file for file in ckpt_list if prefix in file and "checkpoint_" in file]
    num_ckpts = hparams.num_checkpoints
    if len(ckpt_list) <= num_ckpts*2:
        return

    ckpt_list.sort(key=lambda x: int(x.replace(".pth", "").split("_")[-2]))
    num_elim = len(ckpt_list) - num_ckpts
    elim_ckpt = ckpt_list[:num_elim]
    for ckpt in elim_ckpt:
        ckpt_path = os.path.join(checkpoint_dir, ckpt)
        os.remove(ckpt_path)
        print("Deleted", ckpt_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

class GenParallel(nn.DataParallel):
    def forward_with_noise(self, noise, x):
        return self.module.forward_with_noise(noise, x)

class DiscParallel(nn.DataParallel):
    def perceptual_forward(self, x):
        return self.module.perceptual_forward(x)


def run():
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_num)
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

    # Dataset and Dataloader setup
    # train_dataset = Dataset('train_combined_large')
    # test_dataset = Dataset('val_combined_large')
    # train_dataset = Dataset('train_data_large_yen_3_swap')
    # test_dataset = Dataset('val_data_large_yen_3_swap')
    train_dataset = Dataset('train_data')
    test_dataset = Dataset('val_data')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True,
        num_workers=hparams.num_workers, drop_last=False)

    # TODO: uncomment this
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=1, drop_last=False)

    device = torch.device("cuda" if use_cuda else "cpu")

    # TODO: uncomment this
    syncnet = None
    audio_encoder = None
    if args.syncnet_checkpoint_path is not None:
        # syncnet = nn.DataParallel(SyncNet()).to(device)
        syncnet = SyncNet().to(device)
        # syncnet = SyncNet96().to(device)
        load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True,
                                    overwrite_global_states=False)

        audio_encoder = syncnet.audio_encoder
        syncnet = nn.DataParallel(syncnet).to(device)

        for p in syncnet.parameters():
            p.requires_grad = False

     # Model
    model = Wav2Lip(audio_encoder).to(device)
    model.freeze_audio_encoder()
    disc = Wav2Lip_disc_qual().to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters())))
    print('total trainable params {}'.format(sum(p.numel() for p in disc.parameters())))
    print('total trainable params {}'.format(sum(p.numel() for p in syncnet.parameters())))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam([p for p in disc.parameters() if p.requires_grad],
                           lr=hparams.disc_initial_learning_rate, betas=(0.5, 0.999))

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)

    if args.disc_checkpoint_path is not None:
        load_checkpoint(args.disc_checkpoint_path, disc, disc_optimizer,
                                reset_optimizer=False, overwrite_global_states=False)
    model.freeze_audio_encoder()
    model = nn.DataParallel(model).to(device)
    disc = DiscParallel(disc).to(device)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    # Train!
    # hparams.set_hparam('syncnet_wt', 0.03)
    train(device, model, disc, train_data_loader, test_data_loader, optimizer, disc_optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.checkpoint_interval,
          nepochs=hparams.nepochs,
          log_interval=hparams.log_interval,syncnet=syncnet)

def main():
    """Assume Single Node Multi GPUs Training Only"""
    # assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    run()


if __name__ == "__main__":
    main()
