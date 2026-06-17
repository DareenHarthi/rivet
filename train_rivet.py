import os
import json
import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import math
import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler
)

# ECAPA-TDNN
from ecapa import ECAPA_TDNN

# VITS components
from models import SynthesizerTrn, MultiPeriodDiscriminator, SpeakerFlow
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss

# Other utilities
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from text.symbols import symbols

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '80004'

    hps = utils.get_hparams()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    # torch.autograd.set_detect_anomaly(True)
    # ========== Data Loading ==========
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)

    # Use speaker-balanced sampler: 64 speakers � 2 utterances = 128 batch size
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32,300,400,500,600,700,800,900,1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
                              collate_fn=collate_fn, batch_sampler=train_sampler)

    # ========== Model Initialization ==========
    spec_channels = hps.data.filter_length // 2 + 1

    # 1. ECAPA-TDNN (Speaker Encoder)
    net_ecapa = ECAPA_TDNN(input_size=spec_channels).cuda(rank)

    # 2. VITS (Generator and Discriminator)
    net_g = SynthesizerTrn(
        len(symbols),
        spec_channels,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    ).cuda(rank)

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

    # # 3. CNF (Continuous Normalizing Flow)
    net_cnf = SpeakerFlow( 512, 512, 5, 1, 4, gin_channels=512).cuda(rank)

    # ========== Optimizers (before DDP wrapping) ==========
    # Get noise model parameter IDs to exclude from main optimizer
    noise_params_ids = set()
    noise_params_ids.update(id(p) for p in net_ecapa.age_noise_model.parameters())
    noise_params_ids.update(id(p) for p in net_ecapa.gender_noise_model.parameters())

    # Main optimizer (excluding noise matrices - they need their own optimizer)
    ecapa_params_no_noise = [p for p in net_ecapa.parameters() if id(p) not in noise_params_ids]

    optim_g = torch.optim.AdamW(
        list(net_g.parameters()) + list(net_cnf.parameters()) + ecapa_params_no_noise,
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )

    # Separate optimizer for noise matrices (following reference implementation)
    # Uses SGD with no momentum and weight_decay=0 for stability
    optim_noise = torch.optim.SGD(
        list(net_ecapa.age_noise_model.parameters()) +
        list(net_ecapa.gender_noise_model.parameters()),
        lr=hps.train.learning_rate,
        momentum=0,
        weight_decay=0
    )

    # VITS Discriminator optimizer
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps
    )

    # ========== Wrap with DDP ==========
    # Set find_unused_parameters=False for models called multiple times to avoid DDP error
    net_ecapa = DDP(net_ecapa, device_ids=[rank], find_unused_parameters=False, broadcast_buffers=False)
    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=False, broadcast_buffers=False)
    net_d = DDP(net_d, device_ids=[rank], broadcast_buffers=False)
    net_cnf = DDP(net_cnf, device_ids=[rank], find_unused_parameters=False, broadcast_buffers=False)
    # net_cnf = DDP(net_cnf, device_ids=[rank], find_unused_parameters=True)
    # _, _, _, epoch_str = utils.load_checkpoint(
    #     utils.latest_checkpoint_path('logs/ecapa_bce/', "ECAPA_6000.pth"),
    #     net_ecapa, None)
    # ========== Load Checkpoints ==========
    try:
        # Load ECAPA checkpoint
        utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "ECAPA_*.pth"),
            net_ecapa,
            optim_g
        )


        # Load VITS Generator checkpoint
        utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
            net_g,
            optim_g
        )

        # Load VITS Discriminator checkpoint
        utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
            net_d,
            optim_d
        )

        # Load noise matrix optimizer checkpoint
        try:
            utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "NOISE_*.pth"),
                None,  # No model to load, just optimizer
                optim_noise
            )
        except:
            if rank == 0:
                logger.info("No noise optimizer checkpoint found, starting from scratch")

        # # Load CNF checkpoint
        # utils.load_checkpoint(
        #     utils.latest_checkpoint_path(hps.model_dir, "CNF_*.pth"),
        #     net_cnf,
        #     optim_g
        # )
        epoch_str = 1
        global_step = 0

        if rank == 0:
            logger.info(f"Loaded checkpoints from epoch {epoch_str}, global step {global_step}")

    except:
        epoch_str = 1
        global_step = 0
        if rank == 0:
            logger.info("Starting training from scratch")
            

    # ========== Schedulers ==========
    # scheduler_ecapa = torch.optim.lr_scheduler.ExponentialLR(
    #     optim_ecapa, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    # scheduler_cnf = torch.optim.lr_scheduler.ExponentialLR(
    #     optim_cnf, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    # ========== Training Loop ==========
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train(rank, epoch, hps,
                  net_ecapa, net_cnf, net_g,  net_d,
                  optim_g, optim_d, optim_noise, scheduler_g, scheduler_d,
                  scaler, train_loader, logger, writer)
        else:
            train(rank, epoch, hps,
                  net_ecapa, net_cnf, net_g, net_d,
                  optim_g, optim_d, optim_noise, scheduler_g, scheduler_d,
                  scaler, train_loader, None, None)

        # Step all schedulers (note: no scheduler for noise optimizer, following reference implementation)
        # scheduler_ecapa.step()
        scheduler_g.step()
        scheduler_d.step()
        # scheduler_cnf.step()


def train(rank, epoch, hps,
          net_ecapa, net_cnf, net_g, net_d,
          optim_g, optim_d, optim_noise,
          scheduler_g, scheduler_d,
          scaler, train_loader, logger, writer):

    # Set epoch for sampler
    if hasattr(train_loader.batch_sampler, 'set_epoch'):
        train_loader.batch_sampler.set_epoch(epoch)

    global global_step

    # Set models to train mode
    net_ecapa.train()
    net_g.train()
    net_d.train()
    net_cnf.train()
    for batch_idx, (x, x_lengths, spec, spec_aug, spec_lengths, y, y_lengths, _, age, sex) in enumerate(train_loader):
        # Clear gradients at the start of each iteration
        optim_g.zero_grad()
        optim_d.zero_grad()
        optim_noise.zero_grad()

        # Move data to GPU
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        spec_aug = spec_aug.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        age = age.cuda(rank, non_blocking=True)
        sex = sex.cuda(rank, non_blocking=True)
        # print("Max sex:", torch.max(sex))
        # print("Min sex:", torch.min(sex))
        # print("Max age:", torch.max(age))
        # print("Min age:", torch.min(age))

        with autocast(enabled=hps.train.fp16_run):
            # ========== 1. ECAPA-TDNN Forward (Speaker Embedding) ==========
            # Get embeddings from clean spectrogram (weak view)
            spec_t = torch.transpose(spec, 1, 2)
            embeddings, age_pred, gender_pred = net_ecapa(spec_t)

            # Second view: embeddings from augmented spectrogram (strong augmentation)
            spec_aug_t = torch.transpose(spec_aug, 1, 2)
            embeddings_s, _, _ = net_ecapa(spec_aug_t)

            # ========== 2. CNF Forward ==========
            # Weak view (clean spec embeddings)
            z_g_w, logdet_w = net_cnf(embeddings.unsqueeze(-1), age, sex)

            # Strong view (augmented spec embeddings)
            z_g_s, logdet_s = net_cnf(embeddings_s.unsqueeze(-1), age, sex)

     

            # ========== 3. VITS Generator Forward ==========
            y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
            (z, z_p, m_p, logs_p, m_q, logs_q), g = net_g(x, x_lengths, spec, spec_lengths, sid=None, embeddings=embeddings)
            
            # ========== 4. VITS Idempotent ==========
            # with torch.no_grad():
            seg_frames = hps.train.segment_size // hps.data.hop_length  # 32 for 8192//256

            z_seg = commons.slice_segments(z, ids_slice, seg_frames).detach()
            y_hat2 = net_g.module.dec(z_seg , g=embeddings.unsqueeze(-1).detach())
 
            spec_hat = spectrogram_torch(
                y_hat2.squeeze(1),
                hps.data.filter_length,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                center=False
            )

            #   spec_hat_detached = torch.empty_like(spec_hat).copy_(spec_hat) 
            y_hat_lengths = torch.full(
            (z_seg.size(0), ),
                hps.train.segment_size,
                device=z.device,
                dtype=torch.long
            )

            spec_hat_lengths = (y_hat_lengths // hps.data.hop_length).long()
            spec_hat_t = torch.transpose(spec_hat, 1, 2)
            embeddings_idem,  _, _, _ = net_ecapa(spec_hat_t)
            z_idem = net_g(x, x_lengths, spec_hat, spec_hat_lengths, embeddings=embeddings_idem, return_full_spec=True)

    
            # ===================================================
            

            # Mel spectrograms for discriminator
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
   

            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
        with autocast(enabled=False):
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc

        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)


        with autocast(enabled=hps.train.fp16_run):
        # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):

                # Generator losses
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)

                # ========== ECAPA-TDNN Losses ==========
                # Get logits for noisy label training from weak/strong augmented embeddings
                age_logits_w = net_ecapa.module.age_head(embeddings.float())
                age_logits_s = net_ecapa.module.age_head(embeddings_s.float())
                gender_logits_w = net_ecapa.module.gender_head(embeddings.float())
                gender_logits_s = net_ecapa.module.gender_head(embeddings_s.float())

                loss_age = net_ecapa.module.train_step_noisy(
                    age_logits_w, age_logits_s, age,
                    average_entropy_loss=True, label_type="age", num_classes=8
                )
                loss_gender = net_ecapa.module.train_step_noisy(
                    gender_logits_w, gender_logits_s, sex,
                    average_entropy_loss=True, label_type="gender", num_classes=2
                )

                # Compute accuracies for logging
                age_acc = net_ecapa.module._compute_accuracy(age_logits_w, age)
                gender_acc = net_ecapa.module._compute_accuracy(gender_logits_w, sex)

                # Combined ECAPA loss
                loss_ecapa = loss_age + loss_gender

                # ========== CNF Loss ==========
                loss_cnf_mse = F.mse_loss(z_g_w, z_g_s)

                loss_cnf, _ = net_cnf.module.train_step(
                    embeddings.float(), age, sex
                )
                loss_cnf = loss_cnf  + loss_cnf_mse

                # Combined generator loss
                # seg_frames = hps.train.segment_size // hps.data.hop_length  # 32 for 8192//256

                # z_seg = commons.slice_segments(z, ids_slice, seg_frames)
                # print("Z seg shape:", z_seg.shape)
                # print("Z idem shape:", z_idem.shape)
                loss_idem = F.mse_loss(z_seg.detach(), z_idem) *2
                
                loss_sid_idem = F.mse_loss(embeddings.detach(), embeddings_idem) *2
                
                loss_g = loss_gen + loss_fm + loss_mel + loss_kl + loss_dur

                # Total generator + CNF loss
                loss_total = loss_g + loss_ecapa + loss_cnf + loss_idem + loss_sid_idem # Weight CNF loss

        # ========== Backward: Generator + ECAPA + CNF ==========
        scaler.scale(loss_total).backward()

        scaler.unscale_(optim_g)

        # Get ECAPA parameters excluding noise models (consistent with optimizer setup)
        noise_params_ids_local = set()
        noise_params_ids_local.update(id(p) for p in net_ecapa.module.age_noise_model.parameters())
        noise_params_ids_local.update(id(p) for p in net_ecapa.module.gender_noise_model.parameters())
        ecapa_params_no_noise_for_clip = [p for p in net_ecapa.parameters() if id(p) not in noise_params_ids_local]

        grad_norm_g = commons.clip_grad_value_(
            list(net_g.parameters()) + list(net_cnf.parameters()) +
            ecapa_params_no_noise_for_clip, 5.0)

        scaler.step(optim_g)

        # Step noise optimizer separately (following reference implementation)
        scaler.unscale_(optim_noise)
        scaler.step(optim_noise)

        scaler.update()
        del y_hat, y_hat_mel, spec_hat, z, z_g_w

        torch.cuda.empty_cache()

        # ========== Logging ==========
        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr_g = optim_g.param_groups[0]['lr']

                logger.info('Train Epoch: {} [{:.0f}%] Step: {}'.format(
                    epoch, 100. * batch_idx / len(train_loader), global_step))

                logger.info('ECAPA - Loss: {:.4f}, Age Acc: {:.2f}, Gender Acc: {:.2f}'.format(
                    loss_ecapa.item(), age_acc.item(), gender_acc.item()))

                logger.info('ECAPA Losses - Age: {:.4f}, Gender: {:.4f}'.format(
                    loss_age.item(), loss_gender.item()))

                logger.info('VITS - Gen Loss: {:.4f}, Disc Loss: {:.4f}, Mel: {:.4f}, KL: {:.4f}'.format(
                    loss_gen.item(), loss_disc.item(), loss_mel.item(), loss_kl.item()))

                logger.info('CNF - Loss: {:.4f}, MSE Loss: {:.4f}'.format(loss_cnf.item(), loss_cnf_mse.item()))


                logger.info('Idem - Loss: {:.4f}'.format(loss_idem.item()))
                logger.info('Idem SID - Loss: {:.4f}'.format(loss_sid_idem.item()))
                
                



            # ========== Save Checkpoints ==========
            if global_step % hps.train.eval_interval == 0:
                utils.save_checkpoint(
                    net_ecapa, optim_g, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir, "ECAPA_{}.pth".format(global_step)))

                utils.save_checkpoint(
                    net_g, optim_g, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))

                utils.save_checkpoint(
                    net_d, optim_d, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))

                utils.save_checkpoint(
                    net_cnf, optim_g, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir, "CNF_{}.pth".format(global_step)))

                # Save noise optimizer state (noise matrices themselves are saved with net_ecapa)
                torch.save({
                    'optimizer': optim_noise.state_dict(),
                    'learning_rate': hps.train.learning_rate,
                    'epoch': epoch
                }, os.path.join(hps.model_dir, "NOISE_{}.pth".format(global_step)))

        global_step += 1

    if rank == 0:
        logger.info('====> Epoch: {} completed'.format(epoch))


if __name__ == "__main__":
    main()

