# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from tqdm import tqdm

class FlowEuler:
    def __init__(self, model_fn, condition, uncondition, cfg_scale, model_kwargs):
        self.model = model_fn
        self.condition = condition
        self.uncondition = uncondition
        self.cfg_scale = cfg_scale
        self.model_kwargs = model_kwargs
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)

    def sample(self, latents, steps=28, start_step=None,  cond_per_step=None, disable_tqdm=False):
        timesteps = None
       
        device = self.condition.device
        timesteps, _ = retrieve_timesteps(self.scheduler, steps, device, timesteps=timesteps)

        timesteps = timesteps[-start_step:]

        do_classifier_free_guidance = False if self.cfg_scale == 1 else True

        condition = self.condition
        uncondition = self.uncondition

        for i, t in tqdm(list(enumerate(timesteps)), disable=disable_tqdm):
            
            # Condition Steering
            if cond_per_step is not None: 
                if do_classifier_free_guidance: uncondition = cond_per_step[i]
                else: condition = cond_per_step[i]
            
            # CFG
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([uncondition, condition], dim=0)
            else :
                prompt_embeds = condition

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = self.model(
                latent_model_input,
                timestep,
                prompt_embeds,
                **self.model_kwargs,
            )

            if isinstance(noise_pred, Transformer2DModelOutput):
                noise_pred = noise_pred[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        return latents
