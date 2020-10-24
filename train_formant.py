# Copyright 2019-2020 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import json
from os import terminal_size
import pdb
import torch.utils.data
from torchvision.utils import save_image
from net_formant import *
import os
import utils
from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR
from custom_adam import LREQAdam
from dataloader_ecog import *
from tqdm import tqdm
from dlutils.pytorch import count_parameters
import dlutils.pytorch.count_parameters as count_param_override
from tracker import LossTracker
from model_formant import Model
from launcher import run
from defaults import get_cfg_defaults
import lod_driver
from PIL import Image
import numpy as np
from torch import autograd
from ECoGDataSet import concate_batch
from formant_systh import save_sample


def train(cfg, logger, local_rank, world_size, distributed):
    torch.cuda.set_device(local_rank)
    model = Model(
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
        ecog_encoder_name=cfg.MODEL.MAPPING_FROM_ECOG,
        spec_chans = cfg.DATASET.SPEC_CHANS,
        n_formants = cfg.MODEL.N_FORMANTS,
        n_formants_noise = cfg.MODEL.N_FORMANTS_NOISE,
        n_formants_ecog = cfg.MODEL.N_FORMANTS_ECOG,
        wavebased = cfg.MODEL.WAVE_BASED,
        n_fft=cfg.MODEL.N_FFT,
        noise_db=cfg.MODEL.NOISE_DB,
        max_db=cfg.MODEL.MAX_DB,
        with_ecog = cfg.MODEL.ECOG,
        hidden_dim=cfg.MODEL.TRANSFORMER.HIDDEN_DIM,
        dim_feedforward=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
        encoder_only=cfg.MODEL.TRANSFORMER.ENCODER_ONLY,
        attentional_mask=cfg.MODEL.TRANSFORMER.ATTENTIONAL_MASK,
        n_heads = cfg.MODEL.TRANSFORMER.N_HEADS,
        non_local = cfg.MODEL.TRANSFORMER.NON_LOCAL,
        do_mel_guide = cfg.MODEL.DO_MEL_GUIDE,
        noise_from_data = cfg.MODEL.BGNOISE_FROMDATA,
        specsup=cfg.FINETUNE.SPECSUP,
        power_synth = cfg.MODEL.POWER_SYNTH,
    )
    model.cuda(local_rank)
    model.train()

    model_s = Model(
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
        ecog_encoder_name=cfg.MODEL.MAPPING_FROM_ECOG,
        spec_chans = cfg.DATASET.SPEC_CHANS,
        n_formants = cfg.MODEL.N_FORMANTS,
        n_formants_noise = cfg.MODEL.N_FORMANTS_NOISE,
        n_formants_ecog = cfg.MODEL.N_FORMANTS_ECOG,
        wavebased = cfg.MODEL.WAVE_BASED,
        n_fft=cfg.MODEL.N_FFT,
        noise_db=cfg.MODEL.NOISE_DB,
        max_db=cfg.MODEL.MAX_DB,
        with_ecog = cfg.MODEL.ECOG,
        hidden_dim=cfg.MODEL.TRANSFORMER.HIDDEN_DIM,
        dim_feedforward=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
        encoder_only=cfg.MODEL.TRANSFORMER.ENCODER_ONLY,
        attentional_mask=cfg.MODEL.TRANSFORMER.ATTENTIONAL_MASK,
        n_heads = cfg.MODEL.TRANSFORMER.N_HEADS,
        non_local = cfg.MODEL.TRANSFORMER.NON_LOCAL,
        do_mel_guide = cfg.MODEL.DO_MEL_GUIDE,
        noise_from_data = cfg.MODEL.BGNOISE_FROMDATA,
        specsup=cfg.FINETUNE.SPECSUP,
        power_synth = cfg.MODEL.POWER_SYNTH,
    )
    model_s.cuda(local_rank)
    model_s.eval()
    model_s.requires_grad_(False)
    # print(model)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            bucket_cap_mb=25,
            find_unused_parameters=True)
        model.device_ids = None
        decoder = model.module.decoder
        encoder = model.module.encoder
        if hasattr(model.module,'ecog_encoder'):
            ecog_encoder = model.module.ecog_encoder
        if hasattr(model.module,'decoder_mel'):
            decoder_mel = model.module.decoder_mel
    else:
        decoder = model.decoder
        encoder = model.encoder
        if hasattr(model,'ecog_encoder'):
            ecog_encoder = model.ecog_encoder
        if hasattr(model,'decoder_mel'):
            decoder_mel = model.decoder_mel

    count_param_override.print = lambda a: logger.info(a)

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    if cfg.MODEL.ECOG:
        if cfg.MODEL.SUPLOSS_ON_ECOGF:
            optimizer = LREQAdam([
                {'params': ecog_encoder.parameters()}
            ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)
        else:
            optimizer = LREQAdam([
                {'params': ecog_encoder.parameters()},
                {'params': decoder.parameters()},
            ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)
    
    else:
        if cfg.MODEL.DO_MEL_GUIDE:
            optimizer = LREQAdam([
                {'params': encoder.parameters()},
                {'params': decoder.parameters()},
                {'params': decoder_mel.parameters()},
            ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)
        else:
            optimizer = LREQAdam([
                {'params': encoder.parameters()},
                {'params': decoder.parameters()}
            ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    scheduler = ComboMultiStepLR(optimizers=
                                 {'optimizer': optimizer},
                                 milestones=cfg.TRAIN.LEARNING_DECAY_STEPS,
                                 gamma=cfg.TRAIN.LEARNING_DECAY_RATE,
                                 reference_batch_size=32, base_lr=cfg.TRAIN.LEARNING_RATES)
    model_dict = {
        'encoder': encoder,
        'generator': decoder,
    }
    if hasattr(model,'ecog_encoder'):
        model_dict['ecog_encoder'] = ecog_encoder
    if hasattr(model,'decoder_mel'):
        model_dict['decoder_mel'] = decoder_mel
    if local_rank == 0:
        model_dict['encoder_s'] = model_s.encoder
        model_dict['generator_s'] = model_s.decoder
        if hasattr(model_s,'ecog_encoder'):
            model_dict['ecog_encoder_s'] = model_s.ecog_encoder
        if hasattr(model_s,'decoder_mel'):
            model_dict['decoder_mel_s'] = model_s.decoder_mel

    tracker = LossTracker(cfg.OUTPUT_DIR)
    tracker_test = LossTracker(cfg.OUTPUT_DIR,test=True)

    auxiliary = {
                'optimizer': optimizer,
                'scheduler': scheduler,
                'tracker': tracker,
                'tracker_test':tracker_test,
                }

    checkpointer = Checkpointer(cfg, 
                                model_dict,
                                auxiliary,
                                logger=logger,
                                save=local_rank == 0)

    extra_checkpoint_data = checkpointer.load(ignore_last_checkpoint=True,ignore_auxiliary=cfg.FINETUNE.FINETUNE,file_name='./training_artifacts/ecog_finetune_3ecogformants_han5_specsup_guidance_hamonicformantsemph/model_epoch23.pth')
    # extra_checkpoint_data = checkpointer.load(ignore_last_checkpoint=False,ignore_auxiliary=cfg.FINETUNE.FINETUNE,file_name='./training_artifacts/debug_f1f2linearmel/model_epoch27.pth')
    # extra_checkpoint_data = checkpointer.load(ignore_last_checkpoint=False,ignore_auxiliary=cfg.FINETUNE.FINETUNE,file_name='./training_artifacts/debug_fitf1f2freqonly/model_epoch28.pth')
    # extra_checkpoint_data = checkpointer.load(ignore_last_checkpoint=False,ignore_auxiliary=cfg.FINETUNE.FINETUNE,file_name='./training_artifacts/debug_fitf1f2freqonly/model_epoch6.pth')
    # extra_checkpoint_data = checkpointer.load(ignore_last_checkpoint=True,ignore_auxiliary=cfg.FINETUNE.FINETUNE,file_name='./training_artifacts/loudnesscomp_han5/model_epoch50.pth')
    # extra_checkpoint_data = checkpointer.load(ignore_last_checkpoint=False,ignore_auxiliary=cfg.FINETUNE.FINETUNE,file_name='./training_artifacts/test_9/model_epoch30.pth')
    logger.info("Starting from epoch: %d" % (scheduler.start_epoch()))

    arguments.update(extra_checkpoint_data)


    with open('train_param.json','r') as rfile:
        param = json.load(rfile)
    # data_param, train_param, test_param = param['Data'], param['Train'], param['Test']
    dataset = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS,param=param)
    dataset_test = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS,train=False,param=param)
    # noise_dist = dataset.noise_dist
    noise_dist = torch.from_numpy(dataset.noise_dist).to('cuda').float()
    if cfg.MODEL.BGNOISE_FROMDATA:
        model_s.noise_dist_init(noise_dist)
        model.noise_dist_init(noise_dist)
    rnd = np.random.RandomState(3456)
    # latents = rnd.randn(len(dataset_test.dataset), cfg.MODEL.LATENT_SPACE_SIZE)
    # samplez = torch.tensor(latents).float().cuda()


    if cfg.DATASET.SAMPLES_PATH:
        path = cfg.DATASET.SAMPLES_PATH
        src = []
        with torch.no_grad():
            for filename in list(os.listdir(path))[:32]:
                img = np.asarray(Image.open(os.path.join(path, filename)))
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                im = img.transpose((2, 0, 1))
                x = torch.tensor(np.asarray(im, dtype=np.float32), requires_grad=True).cuda() / 127.5 - 1.
                if x.shape[0] == 4:
                    x = x[:3]
                src.append(x)
            sample = torch.stack(src)
    else:
        dataset_test.reset(cfg.DATASET.MAX_RESOLUTION_LEVEL, len(dataset_test.dataset))
        sample_dict_test = next(iter(dataset_test.iterator))
        # sample_dict_test = concate_batch(sample_dict_test)
        sample_wave_test = sample_dict_test['wave_re_batch_all'].to('cuda').float()
        if cfg.MODEL.WAVE_BASED:
            sample_spec_test = sample_dict_test['wave_spec_re_batch_all'].to('cuda').float()
            sample_spec_amp_test = sample_dict_test['wave_spec_re_amp_batch_all'].to('cuda').float()
            sample_spec_denoise_test = sample_dict_test['wave_spec_re_denoise_batch_all'].to('cuda').float()
            # sample_spec_test = wave2spec(sample_wave_test,n_fft=cfg.MODEL.N_FFT,noise_db=cfg.MODEL.NOISE_DB,max_db=cfg.MODEL.MAX_DB)
        else:
            sample_spec_test = sample_dict_test['spkr_re_batch_all'].to('cuda').float()
            sample_spec_denoise_test = None#sample_dict_test['wave_spec_re_denoise_batch_all'].to('cuda').float()
        sample_label_test = sample_dict_test['label_batch_all']
        if cfg.MODEL.ECOG:
            ecog_test = [sample_dict_test['ecog_re_batch_all'][i].to('cuda').float() for i in range(len(sample_dict_test['ecog_re_batch_all']))]
            mask_prior_test = [sample_dict_test['mask_all'][i].to('cuda').float() for i in range(len(sample_dict_test['mask_all']))]
            mni_coordinate_test = sample_dict_test['mni_coordinate_all'].to('cuda').float()
        else:
            ecog_test = None
            mask_prior_test = None
            mni_coordinate_test = None
        sample_spec_mel_test = sample_dict_test['spkr_re_batch_all'].to('cuda').float() if cfg.MODEL.DO_MEL_GUIDE else None
        on_stage_test = sample_dict_test['on_stage_re_batch_all'].to('cuda').float()
        on_stage_wider_test = sample_dict_test['on_stage_wider_re_batch_all'].to('cuda').float()
        # sample = next(make_dataloader(cfg, logger, dataset, 32, local_rank))
        # sample = (sample / 127.5 - 1.)
    # import pdb; pdb.set_trace()
    duomask=True
    # model.eval()
    # Lrec = model(sample_spec_test, x_denoise = sample_spec_denoise_test,x_mel = sample_spec_mel_test,ecog=ecog_test if cfg.MODEL.ECOG else None, mask_prior=mask_prior_test if cfg.MODEL.ECOG else None, on_stage = on_stage_test,on_stage_wider = on_stage_wider_test, ae = not cfg.MODEL.ECOG, tracker = tracker_test, encoder_guide=cfg.MODEL.W_SUP,pitch_aug=False,duomask=duomask,mni=mni_coordinate_test,debug = False,x_amp=sample_spec_amp_test,hamonic_bias = False)
    # save_sample(sample_spec_test,ecog_test,mask_prior_test,mni_coordinate_test,encoder,decoder,ecog_encoder if cfg.MODEL.ECOG else None,x_denoise=sample_spec_denoise_test,x_mel = sample_spec_mel_test,decoder_mel=decoder_mel if cfg.MODEL.DO_MEL_GUIDE else None,epoch=0,label=sample_label_test,mode='test',path=cfg.OUTPUT_DIR,tracker = tracker_test,linear=cfg.MODEL.WAVE_BASED,n_fft=cfg.MODEL.N_FFT,duomask=True)
    for epoch in range(cfg.TRAIN.TRAIN_EPOCHS):
        model.train()

        # batches = make_dataloader(cfg, logger, dataset, lod2batch.get_per_GPU_batch_size(), local_rank)
        model.train()
        need_permute = False
        epoch_start_time = time.time()
        i = 0
        for sample_dict_train in tqdm(iter(dataset.iterator)):
            # import pdb; pdb.set_trace()
            # sample_dict_train = concate_batch(sample_dict_train)
            i += 1
            wave_orig = sample_dict_train['wave_re_batch_all'].to('cuda').float()
            if cfg.MODEL.WAVE_BASED:
                # x_orig = wave2spec(wave_orig,n_fft=cfg.MODEL.N_FFT,noise_db=cfg.MODEL.NOISE_DB,max_db=cfg.MODEL.MAX_DB)
                x_orig = sample_dict_train['wave_spec_re_batch_all'].to('cuda').float()
                x_orig_amp = sample_dict_train['wave_spec_re_amp_batch_all'].to('cuda').float()
                x_orig_denoise = sample_dict_train['wave_spec_re_denoise_batch_all'].to('cuda').float()
            else:
                x_orig = sample_dict_train['spkr_re_batch_all'].to('cuda').float()
                x_orig_denoise = None#sample_dict_train['wave_spec_re_denoise_batch_all'].to('cuda').float()

            on_stage = sample_dict_train['on_stage_re_batch_all'].to('cuda').float()
            on_stage_wider = sample_dict_train['on_stage_wider_re_batch_all'].to('cuda').float()
            words = sample_dict_train['word_batch_all'].to('cuda').long()
            words = words.view(words.shape[0]*words.shape[1])
            labels = sample_dict_train['label_batch_all']
            if cfg.MODEL.ECOG:
                ecog = [sample_dict_train['ecog_re_batch_all'][j].to('cuda').float() for j in range(len(sample_dict_train['ecog_re_batch_all']))]
                mask_prior = [sample_dict_train['mask_all'][j].to('cuda').float() for j in range(len(sample_dict_train['mask_all']))]
                mni_coordinate = sample_dict_train['mni_coordinate_all'].to('cuda').float()
            else:
                ecog = None
                mask_prior = None
                mni_coordinate = None
            x = x_orig
            x_mel = sample_dict_train['spkr_re_batch_all'].to('cuda').float() if cfg.MODEL.DO_MEL_GUIDE else None
            # x.requires_grad = True
            # apply_cycle = cfg.MODEL.CYCLE and True
            # apply_w_classifier = cfg.MODEL.W_CLASSIFIER and True
            # apply_gp = True
            # apply_ppl = cfg.MODEL.APPLY_PPL and True
            # apply_ppl_d = cfg.MODEL.APPLY_PPL_D and True
            # apply_encoder_guide = (cfg.FINETUNE.ENCODER_GUIDE or cfg.MODEL.W_SUP) and True
            # apply_sup = cfg.FINETUNE.SPECSUP
            
            if (cfg.MODEL.ECOG):
                optimizer.zero_grad()
                Lrec = model(x, x_denoise = x_orig_denoise,x_mel = x_mel,ecog=ecog, mask_prior=mask_prior, on_stage = on_stage,on_stage_wider = on_stage_wider, ae = False, tracker = tracker, encoder_guide=cfg.MODEL.W_SUP,duomask=duomask,mni=mni_coordinate,x_amp=x_orig_amp)
                (Lrec).backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                Lrec = model(x, x_denoise = x_orig_denoise,x_mel = x_mel,ecog=None, mask_prior=None, on_stage = on_stage,on_stage_wider = on_stage_wider, ae = True, tracker = tracker, encoder_guide=cfg.MODEL.W_SUP,pitch_aug=False,duomask=duomask,mni=mni_coordinate,debug = False,x_amp=x_orig_amp,hamonic_bias = False)#epoch<2)
                (Lrec).backward()
                optimizer.step()

            betta = 0.5 ** (cfg.TRAIN.BATCH_SIZE / (10 * 1000.0))
            model_s.lerp(model, betta,w_classifier = cfg.MODEL.W_CLASSIFIER)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time


        if local_rank == 0:
            print(2**(torch.tanh(model.encoder.formant_bandwitdh_slop)))
            checkpointer.save("model_epoch%d" % epoch)
            model.eval()
            Lrec = model(sample_spec_test, x_denoise = sample_spec_denoise_test,x_mel = sample_spec_mel_test,ecog=ecog_test if cfg.MODEL.ECOG else None, mask_prior=mask_prior_test if cfg.MODEL.ECOG else None, on_stage = on_stage_test,on_stage_wider = on_stage_wider_test, ae = not cfg.MODEL.ECOG, tracker = tracker_test, encoder_guide=cfg.MODEL.W_SUP,pitch_aug=False,duomask=duomask,mni=mni_coordinate_test,debug = False,x_amp=sample_spec_amp_test,hamonic_bias = False)
            save_sample(sample_spec_test,ecog_test,mask_prior_test,mni_coordinate_test,encoder,decoder,ecog_encoder if cfg.MODEL.ECOG else None,x_denoise=sample_spec_denoise_test,x_mel = sample_spec_mel_test,decoder_mel=decoder_mel if cfg.MODEL.DO_MEL_GUIDE else None,epoch=epoch,label=sample_label_test,mode='test',path=cfg.OUTPUT_DIR,tracker = tracker_test,linear=cfg.MODEL.WAVE_BASED,n_fft=cfg.MODEL.N_FFT,duomask=duomask,x_amp=sample_spec_amp_test)
            save_sample(x,ecog,mask_prior,mni_coordinate,encoder,decoder,ecog_encoder if cfg.MODEL.ECOG else None,x_denoise=x_orig_denoise,x_mel = sample_spec_mel_test,decoder_mel=decoder_mel if cfg.MODEL.DO_MEL_GUIDE else None,epoch=epoch,label=labels,mode='train',path=cfg.OUTPUT_DIR,tracker = tracker,linear=cfg.MODEL.WAVE_BASED,n_fft=cfg.MODEL.N_FFT,duomask=duomask,x_amp=x_orig_amp)


if __name__ == "__main__":
    gpu_count = torch.cuda.device_count()
    run(train, get_cfg_defaults(), description='StyleGAN', default_config='configs/ecog_style2.yaml',
        world_size=gpu_count)
