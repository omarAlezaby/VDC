# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import pyrallis
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
import cv2
import numpy as np
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers import FlowMatchEulerDiscreteScheduler
import os
import random
from torchvision.utils import save_image
import logging
import shutil

warnings.filterwarnings("ignore")  # ignore warning

from diffusion.scheduler.flow_euler_inv_sampler import FlowEulerInv
from diffusion.scheduler.flow_euler_vdc_sampler import FlowEuler


from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.model.utils import get_weight_dtype, resize_and_crop_tensor
from diffusion.utils.config import SanaConfig, model_init_config
from diffusion.utils.logger import get_root_logger

# from diffusion.utils.misc import read_config
from tools.download import find_model



def guidance_type_select(default_guidance_type, pag_scale, attn_type):
    guidance_type = default_guidance_type
    if not (pag_scale > 1.0 and attn_type == "linear"):
        guidance_type = "classifier-free"
    elif pag_scale > 1.0 and attn_type == "linear":
        guidance_type = "classifier-free_PAG"
    return guidance_type


def classify_height_width_bin(height: int, width: int, ratios: dict) -> Tuple[int, int]:
    """Returns binned height and width."""
    ar = float(height / width)
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
    default_hw = ratios[closest_ratio]
    return int(default_hw[0]), int(default_hw[1])

@dataclass
class SanaInference(SanaConfig):
    config: Optional[str] = "configs/sana_config/1024ms/Sana_1600M_img1024.yaml"  # config
    model_path: str = field(
        default="output/Sana_D20/SANA.pth", metadata={"help": "Path to the model file (positional)"}
    )
    output: str = "./output"
    bs: int = 1
    image_size: int = 1024
    cfg_scale: float = 4.5
    pag_scale: float = 1.0
    seed: int = 42
    step: int = -1
    custom_image_size: Optional[int] = None
    shield_model_path: str = field(
        default="google/shieldgemma-2b",
        metadata={"help": "The path to shield model, we employ ShieldGemma-2B by default."},
    )


class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
    
class SirenBase(nn.Module):

    def __init__(self, n_input_dims=1, n_output_dims=768, sin_w=1, n_neurons: int = 128, 
                 n_hidden_layers: int = 1, use_skip: bool = False, return_update: bool = False):
        super().__init__()
        layers = [nn.Linear(n_input_dims, n_neurons), Sine(sin_w)]
        for i in range(1,n_hidden_layers+1):
        #    layers.append(nn.Linear(max(1,(i-1)*2)*n_neurons, i*2*n_neurons))
           layers.append(nn.Linear(n_neurons, n_neurons))
           layers.append(Sine(sin_w))

        # layers.append(nn.Linear(n_hidden_layers*2*n_neurons, n_output_dims))
        layers.append(nn.Linear(n_neurons, n_output_dims))

        self.model = nn.Sequential(*layers)

        self.use_skip = use_skip
        self.return_update = return_update
       

    def forward(self, inp):
        output = self.model(inp)

        if self.use_skip :
            update = output
            output = update + inp # global skip connectionn
            if self.return_update:
                return update, output
    
        return output


class SanaINRPipeline(nn.Module):
    def __init__(
        self,
        config: Optional[str] = "configs/sana_config/1024ms/Sana_1600M_img1024.yaml",
    ):
        super().__init__()
        config = pyrallis.load(SanaInference, open(config))
        self.args = self.config = config

        # set some hyper-parameters
        self.image_size = self.config.model.image_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger = get_root_logger()
        self.logger = logger
        self.progress_fn = lambda progress, desc: None

        self.latent_size = self.image_size // config.vae.vae_downsample_rate
        self.max_sequence_length = config.text_encoder.model_max_length
        self.flow_shift = config.scheduler.flow_shift
        guidance_type = "classifier-free"

        weight_dtype = get_weight_dtype(config.model.mixed_precision)
        self.weight_dtype = weight_dtype
        self.vae_dtype = get_weight_dtype(config.vae.weight_dtype)

        self.base_ratios = eval(f"ASPECT_RATIO_{self.image_size}_TEST")
        self.vis_sampler = self.config.scheduler.vis_sampler
        # self.vis_sampler = "flow_euler"
        logger.info(f"Sampler {self.vis_sampler}, flow_shift: {self.flow_shift}")
        self.guidance_type = guidance_type_select(guidance_type, self.args.pag_scale, config.model.attn_type)
        logger.info(f"Inference with {self.weight_dtype}, PAG guidance layer: {self.config.model.pag_applied_layers}")

        # 1. build vae and text encoder
        self.vae = self.build_vae(config.vae)
        for j, p in enumerate(self.vae.parameters()):
            p.requires_grad_(False)
        tokenizer, text_encoder = self.build_text_encoder(config.text_encoder)

        # 2. build Sana model
        self.model = self.build_sana_model(config).to(self.device)
        for j, p in enumerate(self.model.parameters()):
            p.requires_grad_(False)


        # # 3. pre-compute null embedding
        with torch.no_grad():
            self.null_caption_token = tokenizer(
                "", max_length=self.max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(self.device)
            self.null_caption_embs = text_encoder(self.null_caption_token.input_ids, self.null_caption_token.attention_mask)[
                0
            ]

    def build_vae(self, config):
        vae = get_vae(config.vae_type, config.vae_pretrained, self.device).to(self.vae_dtype)
        return vae

    def build_text_encoder(self, config):
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder_name, device=self.device)
        return tokenizer, text_encoder

    def build_sana_model(self, config):
        # model setting
        model_kwargs = model_init_config(config, latent_size=self.latent_size)
        model = build_model(
            config.model.model,
            use_fp32_attention=config.model.get("fp32_attention", False) and config.model.mixed_precision != "bf16",
            use_grad_checkpoint=True,
            **model_kwargs,
        )
        self.logger.info(f"use_fp32_attention: {model.fp32_attention}")
        self.logger.info(
            f"{model.__class__.__name__}:{config.model.model},"
            f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}"
        )
        return model

    def from_pretrained(self, model_path):
        state_dict = find_model(model_path)
        state_dict = state_dict.get("state_dict", state_dict)
        if "pos_embed" in state_dict:
            del state_dict["pos_embed"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(self.weight_dtype)

        self.logger.info("Generating sample from ckpt: %s" % model_path)
        self.logger.warning(f"Missing keys: {missing}")
        self.logger.warning(f"Unexpected keys: {unexpected}")

    def register_progress_bar(self, progress_fn=None):
        self.progress_fn = progress_fn if progress_fn is not None else self.progress_fn

    def aug_img(self, inp, gt, device, color_aug=True, affine_aug=True):

        # Random horizontal flipping
        if random.random() > 0.5:
            inp = TF.hflip(inp)
            gt = TF.hflip(gt)

        # Random vertical flipping
        if random.random() > 0.5:
            inp = TF.vflip(inp)
            gt = TF.vflip(gt)
        if random.random() > .5:
            angel = random.randint(-5, -5)
            inp = TF.rotate(inp, angle=angel)
            gt = TF.rotate(gt, angle=angel)

                # Random crop

        if affine_aug:
            if random.random() > .5:
                i, j, h, w = T.RandomResizedCrop.get_params(
                    inp, scale=(1.0,1.1), ratio=(1,1))
                # inp = TF.crop(inp, i, j, h, w)
                # gt = TF.crop(gt, i, j, h, w)
                inp = TF.resized_crop(inp, i, j, h, w, size=(512, 512))
                gt = TF.resized_crop(gt, i, j, h, w, size=(512, 512))

            if random.random() > .5:
                angle, translations, scale, shear = T.RandomAffine.get_params(translate= (.1,.1), degrees=(0,0), scale_ranges= None, shears=None,
                                                                            img_size=(512, 512))
                # inp = TF.crop(inp, i, j, h, w)
                # gt = TF.crop(gt, i, j, h, w)
                inp = TF.affine(inp, angle, translations, scale, shear)
                gt = TF.affine(gt, angle, translations, scale, shear)


        if color_aug and random.random() > .5:
            
            fn_idx, b, c, s, h = T.ColorJitter.get_params(brightness = (.8,1.2), contrast = (.8,1.2), saturation = (.8,1.2), hue= (-.2,.2))
            for fn_id in fn_idx:
                if fn_id == 0 and b is not None:
                    inp = TF.adjust_brightness(inp, b)
                    gt = TF.adjust_brightness(gt, b)
                elif fn_id == 1 and c is not None:
                    inp = TF.adjust_contrast(inp, c)
                    gt = TF.adjust_contrast(gt, c)
                elif fn_id == 2 and s is not None:
                    inp = TF.adjust_saturation(inp, s)
                    gt = TF.adjust_saturation(gt, s)
                elif fn_id == 3 and h is not None:
                    inp = TF.adjust_hue(inp, h)
                    gt = TF.adjust_hue(gt, h)

        # Transform to tensor
        inp = TF.to_tensor(inp)
        gt = TF.to_tensor(gt)
        inp = TF.normalize(inp, [0.5], [0.5])
        gt = TF.normalize(gt, [0.5], [0.5])

        return inp.unsqueeze(0).to(device), gt.unsqueeze(0).to(device)
    

    def get_img(self, img_path):
        img = Image.open(img_path)
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB")),
                # T.CenterCrop(size=(512,512)),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )

        return transform(img).unsqueeze(0).to(self.device)
    

    
    def optimize_noise(self, z_enc, z_ref, steps, start_step, condition, uncondition, model_kwargs, lr, itrs=20):
        scaler = torch.amp.GradScaler("cuda")
        flow_solver = FlowEuler(
                    self.model,
                    condition=condition,
                    uncondition=uncondition,
                    cfg_scale=1,
                    model_kwargs=model_kwargs,
                )
        
        z_enc.requires_grad = True
        adam = torch.optim.AdamW(params = [z_enc], lr = lr)
        loss_fn = torch.nn.functional.mse_loss

        pbar = tqdm(range(itrs))
        for i in pbar:
            # decode it
            sample = flow_solver.sample(
                        z_enc,
                        steps=steps,
                        start_step= start_step,
                        refinment=False,
                    )
            loss = loss_fn(sample, z_ref, reduction = 'mean')


            scaler.scale(loss).backward()
            scaler.step(adam)
            scaler.update()
            adam.zero_grad()
            logging.info(f"Noise Optim, itr {i} Loss: {loss.item()}")

        return z_enc
    
    def input_mapping(self, x, B):
        if B is None:
            return x
        else:
            x_proj = (2. * np.pi * x) @ B.t()
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
    def optimize_INR(self, before_imgs, ref_imgs, steps, start_step, uncondition, model_kwargs, lr, scale, itrs, batch_size):
        flow_solver = FlowEuler(
                    self.model,
                    condition=uncondition,
                    uncondition=uncondition,
                    cfg_scale=scale,
                    model_kwargs=model_kwargs,
                )
        flow_inv_solver = FlowEulerInv(
                    self.model,
                    condition=self.null_caption_embs,
                    uncondition=self.null_caption_embs,
                    cfg_scale=1,
                    model_kwargs=model_kwargs,
                )
        scaler = torch.amp.GradScaler("cuda")
        
        pos_enc = torch.arange(1,self.max_sequence_length+1, step=1).to(self.device)/self.max_sequence_length
        mapping_size = 256
        B_gauss = torch.randn((mapping_size, 1)).to(self.device) * 10
        mapp = self.input_mapping(pos_enc.unsqueeze(1),B_gauss)

        mapping_nets = []
        for _ in range(start_step):
            mapping_net = SirenBase(n_input_dims=2*mapping_size, n_output_dims=uncondition.shape[-1]).to(self.device)
            mapping_net.train()
            mapping_nets.append(mapping_net)

        adam = torch.optim.Adam([{'params':mapping_net.parameters()} for mapping_net in mapping_nets], lr = lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=itrs, eta_min=lr * .2)
        loss_fn = torch.nn.functional.mse_loss

        pbar = tqdm(range(itrs))        

        for iii in pbar:
            loss_total=0
            for ba in range(batch_size):
                data_num = random.sample(range(len(ref_imgs)), 1)[0]
                # decode it
                out_conds = []
                for mapping_net in mapping_nets:
                    out_conds.append(mapping_net(mapp).unsqueeze(0))

                aug_before, aug_ref = self.aug_img(before_imgs[data_num], ref_imgs[data_num], uncondition.device)

                before_latent = vae_encode(self.config.vae.vae_type, self.vae, aug_before, self.config.vae.sample_posterior, self.device)
                ref_latent = vae_encode(self.config.vae.vae_type, self.vae, aug_ref, self.config.vae.sample_posterior, self.device)


                before_enc = flow_inv_solver.sample(
                    before_latent,
                    steps=steps,
                    start_step= start_step,
                )


                sample = flow_solver.sample(
                            before_enc,
                            steps=steps,
                            start_step= start_step,
                            cond_per_step=out_conds
                        )
                sample_out = vae_decode(self.config.vae.vae_type, self.vae, sample)
                loss = loss_fn(sample, ref_latent, reduction = 'mean')/batch_size
                loss += loss_fn(sample_out, aug_ref, reduction = 'mean')/batch_size

                loss_total+=loss.item()


                scaler.scale(loss).backward()
            scaler.step(adam)
            scaler.update()
            sched.step()
            adam.zero_grad()
            logging.info(f"Cond Optim, itr {iii} Loss: {loss_total}")
            
        out_conds = []
        for mapping_net in mapping_nets:
            out_conds.append(mapping_net(mapp).unsqueeze(0))
        return out_conds
    
    # @torch.inference_mode()
    def forward(
        self,
        height=512,
        width=512,
        num_inference_steps=100,
        guidance_scale=7.0,
        pag_guidance_scale=1.0,
        before_path=None,  # optimization visual example input
        after_path=None, # optimization visual example reference
        inp_path=None, # inference images
        out_dir=None,
        use_resolution_binning=True,
        sampler="flow_euler",
        strenght=.1,
        opt_itrs=200,
        opt_lr=.005,
        optimize_noise=False,
        num_train_samples=1,
        batch_size=4,
        out_sampels=None,
        optim_conds_path=None,
    ):
        
        logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f'{out_dir}/output.log'),
            logging.StreamHandler()])

            
        if sampler != None:
            self.vis_sampler = sampler
        self.ori_height, self.ori_width = height, width
        if use_resolution_binning:
            self.height, self.width = classify_height_width_bin(height, width, ratios=self.base_ratios)
        else:
            self.height, self.width = height, width
        self.latent_size_h, self.latent_size_w = (
            self.height // self.config.vae.vae_downsample_rate,
            self.width // self.config.vae.vae_downsample_rate,
        )
        self.guidance_type = guidance_type_select(self.guidance_type, pag_guidance_scale, self.config.model.attn_type)

        _, hw, ar = (
            [],
            torch.tensor([[self.image_size, self.image_size]], dtype=torch.float, device=self.device).repeat(
                1, 1
            ),
            torch.tensor([[1.0]], device=self.device).repeat(1, 1),
        )

        if optim_conds_path is None:
            ref_imgs = []
            before_imgs = []
            if os.path.isdir(before_path):
                imgs_names = os.listdir(before_path)
                imgs_names.sort()
                imgs_names = [f"{before_path}/{imgs_names[i]}" for i in range(num_train_samples)]
                ref_names = os.listdir(after_path)
                ref_names.sort()
                ref_names = [f"{after_path}/{ref_names[i]}" for i in range(num_train_samples)]
            else:
                imgs_names = [before_path]
                ref_names = [after_path]

            for fi, ref_fi in zip(imgs_names, ref_names):
                before_imgs.append(Image.open(f"{fi}").convert("RGB"))
                ref_imgs.append(Image.open(f"{ref_fi}").convert("RGB"))
                
            logging.info(f"train on images: {imgs_names}")


        with torch.no_grad():
 
            # number of inference steps
            start_step= int(num_inference_steps*strenght)
            emb_masks = torch.zeros(1, self.max_sequence_length).to(self.device)
            emb_masks[:,:] = 1
            model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)


 

            # get noising steps
            scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
            timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, self.device, timesteps=None)
            noise_steps = timesteps[-start_step].long()
            del scheduler
            logging.info(f"Number of Utilizied Steps: {noise_steps}")

            with torch.enable_grad():
                if optim_conds_path is None:
                    optim_conds = self.optimize_INR(before_imgs=before_imgs, ref_imgs=ref_imgs, steps=num_inference_steps, start_step=start_step,
                                                        uncondition=self.null_caption_embs, model_kwargs=model_kwargs, lr=opt_lr, scale=guidance_scale,
                                                        itrs=opt_itrs, batch_size=batch_size)
                    np.save(f"{out_dir}/learned_conds.npy", torch.cat(optim_conds).detach().cpu().numpy())
                else:
                    optim_conds= torch.from_numpy(np.load(optim_conds_path)).to(self.device).unsqueeze(1)
                
            with torch.no_grad():
                flow_solver = FlowEuler(
                    self.model,
                    condition=self.null_caption_embs,
                    uncondition=self.null_caption_embs,
                    cfg_scale=guidance_scale,
                    model_kwargs=model_kwargs,
                )
                flow_inverser = FlowEulerInv(
                        self.model,
                        condition=self.null_caption_embs,
                        uncondition=self.null_caption_embs,
                        cfg_scale=1,
                        model_kwargs=model_kwargs,
                    )
                
                samples = {}
                if os.path.isdir(inp_path):
                    inp_files = [f"{inp_path}/{fi}" for fi in os.listdir(inp_path)]
                    if out_sampels is not None:
                        inp_files = inp_files[:out_sampels]
                else :
                    inp_files = [inp_path]
                for fi in inp_files :
                    logging.info(f"inference on image: {fi}")
                    img = self.get_img(fi)
                    latent_inp = vae_encode(self.config.vae.vae_type, self.vae, img, self.config.vae.sample_posterior, self.device)
                    enc_inp = flow_inverser.sample(
                        latent_inp,
                        steps=num_inference_steps,
                        start_step= start_step,
                        disable_tqdm=True,
                    )
                    if optimize_noise:
                        with torch.enable_grad():
                            enc_inp = self.optimize_noise(enc_inp, latent_inp, steps=num_inference_steps, start_step= start_step, itrs=20, 
                                                condition=self.null_caption_embs, uncondition=self.null_caption_embs, model_kwargs=model_kwargs, lr=.001)
                    sample = flow_solver.sample(
                        enc_inp,
                        steps=num_inference_steps,
                        start_step= start_step,
                        cond_per_step=optim_conds,
                    )
                    sample = sample.to(self.vae_dtype)
                    sample = vae_decode(self.config.vae.vae_type, self.vae, sample)
                    if use_resolution_binning:
                        sample = resize_and_crop_tensor(sample, self.ori_width, self.ori_height)
                    samples[fi.rsplit('/',1)[-1]] = sample
                    save_image(sample, f"{out_dir}/{fi.rsplit('/',1)[-1]}", nrow=1, normalize=True, value_range=(-1, 1))
                
                return samples