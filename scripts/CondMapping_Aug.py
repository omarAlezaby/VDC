"""make variations of input image"""

import argparse, os
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from itertools import islice
from einops import rearrange, repeat
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import torch
import torch.nn as nn
import random
import shutil
import logging
import torchvision.transforms.functional as TF
import torchvision.transforms as T


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    logging.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        logging.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logging.info("missing keys:")
        logging.info(m)
    if len(u) > 0 and verbose:
        logging.info("unexpected keys:")
        logging.info(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    logging.info(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def aug_img(inp, gt, device, color_aug=True, affine_aug=True):

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

    # affine augmentation
    if affine_aug:
        if random.random() > .5:
            i, j, h, w = T.RandomResizedCrop.get_params(
                inp, scale=(1.0,1.1), ratio=(1,1))
            inp = TF.resized_crop(inp, i, j, h, w, size=(512, 512))
            gt = TF.resized_crop(gt, i, j, h, w, size=(512, 512))

        if random.random() > .5:
            angle, translations, scale, shear = T.RandomAffine.get_params(translate= (.1,.1), degrees=(0,0), scale_ranges= None, shears=None,
                                                                        img_size=(512, 512))
            inp = TF.affine(inp, angle, translations, scale, shear)
            gt = TF.affine(gt, angle, translations, scale, shear)


    # Color augmentation
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


def optimize_inversion(ref_img, ref_latent, z_enc, t_enc, init_cond, sampler, itrs, uc):
    # make the ddim inverted latent optimizable
    z_enc.requires_grad = True
    scaler = torch.amp.GradScaler("cuda")

    adam = torch.optim.AdamW(params = [z_enc], lr = .01)
    loss_fn = torch.nn.functional.mse_loss

    pbar = tqdm(range(itrs))
    for i in pbar:
        # decode it
        samples = sampler.decode(z_enc, init_cond, t_enc, unconditional_guidance_scale=1,
                                    unconditional_conditioning=uc)
        
        # Optimize the ddim inverted latent to better sample the orginal latent
        loss = loss_fn(samples, ref_latent, reduction = 'mean') 

        scaler.scale(loss).backward()
        scaler.step(adam)
        scaler.update()
        adam.zero_grad()
        logging.info(f"itr {i} Inversion Opt Loss: {loss.item()}")

    return z_enc


class SirenBase(nn.Module):

    def __init__(self, n_input_dims=768, n_output_dims=768, sin_w=1, n_neurons: int = 128, 
                 n_hidden_layers: int = 1, use_skip: bool = False, return_update: bool = False, dropout: float = None):
        super().__init__()
        layers = [nn.Linear(n_input_dims, n_neurons), nn.ReLU()]
        for i in range(n_hidden_layers):
           layers.append(nn.Linear(n_neurons, n_neurons))
           layers.append(nn.ReLU())
           if dropout is not None: layers.append(nn.Dropout(p=dropout))

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


# Postion encoding
def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    

def cond_mapping_perstep_mlp(ref_imgs, before_imgs, t_enc, init_cond, sampler, itrs, scale, uc, batch_size=2):
    
    # Postion encoding using fourier features
    pos_enc = torch.arange(1,init_cond.shape[1]+1, step=1).to(init_cond.device)/init_cond.shape[1]
    mapping_size = 256
    with torch.autocast(device_type="cuda", enabled=False):
        B_gauss = torch.randn((mapping_size, 1)).to(init_cond.device) * 10
        mapp = input_mapping(pos_enc.unsqueeze(1),B_gauss)


    # Condition Generation MLPs
    scaler = torch.amp.GradScaler("cuda")
    mapping_nets = []
    for _ in range(t_enc):
        mapping_net = SirenBase(n_input_dims=2*mapping_size).to(init_cond[0].device)
        mapping_net.train()
        mapping_nets.append(mapping_net)

    # Optimization Parameters
    adam = torch.optim.Adam([{'params':mapping_net.parameters()} for mapping_net in mapping_nets], lr=.005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=itrs, eta_min=.001)
    loss_fn = torch.nn.functional.mse_loss

    pbar = tqdm(range(itrs))
    for iii in pbar:
        total_loss = 0
        # Batch Optimization via Gradient Accumulation 
        for ba in range(batch_size):
            data_num = random.sample(range(len(ref_imgs)),1)[0]

            # Condition generation
            out_conds = []
            with torch.autocast(device_type="cuda", enabled=False):
                for mapping_net in mapping_nets:
                    out_conds.append(mapping_net(mapp).unsqueeze(0))
            
            # Online augmentation
            aug_before, aug_ref = aug_img(before_imgs[data_num], ref_imgs[data_num], init_cond.device)
            before_latent = sampler.model.get_first_stage_encoding(sampler.model.encode_first_stage(aug_before))  # move to latent space
            ref_latent = sampler.model.get_first_stage_encoding(sampler.model.encode_first_stage(aug_ref)).detach()  # move to latent space

            latent_z = sampler.ddim_inverse(before_latent, uc, t_enc)[-1].detach()

            # Condition Steering
            samples = sampler.decode_VDC(latent_z, uc, t_enc, unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc, perstep_cond=out_conds)

            # Condition Optimization
            x_samples = sampler.model.decode_first_stage(samples)
            loss = loss_fn(samples, ref_latent, reduction = 'mean')/batch_size
            loss += loss_fn(x_samples, aug_ref, reduction = 'mean')/batch_size

            total_loss += loss.item()
            
            scaler.scale(loss).backward()
        scaler.step(adam)
        scaler.update()
        sched.step()
        adam.zero_grad()

        logging.info(f"itr {iii} Condition Opt Loss: {total_loss}")

    # Optimized condition generation 
    with torch.autocast(device_type="cuda", enabled=False):
        out_conds = []
        for mapping_net in mapping_nets:
            out_conds.append(mapping_net(mapp).unsqueeze(0))

        return out_conds
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-imgs",
        type=str,
        nargs="?",
        help="path to the input image file (one-shot) or folder of images (multi-shot) used for optimizing the condition",
        default="/home/omar/datasets/Rain100L/Test/resized_rain",
    )

    parser.add_argument(
        "--reference-imgs",
        type=str,
        nargs="?",
        help="path to the reference image file (one-shot) or folder of images (multi-shot) used for optimizing the condition",
        default="/home/omar/datasets/Rain100L/Test/resized_norain",
    )

    parser.add_argument(
        "--inference-imgs",
        type=str,
        nargs="?",
        help="path to the image file or folder of images for inference after optimization",
        default="/home/omar/datasets/Rain100L/Test/resized_rain",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/rain_aug"
    )

    parser.add_argument(
        "--num-train-samples",
        type=int,
        nargs="?",
        help="number of examples to use for condition optimization",
        default=1,
    )

    parser.add_argument(
        "--num-inference-samples",
        type=int,
        nargs="?",
        help="number of images for inference",
        default=0,
    )

    parser.add_argument(
        "--opt_cond_itrs",
        type=int,
        default=200,
        help="number of iteration to optimize the steering condition",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch size for condition optimization",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save inference output.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    
    parser.add_argument(
        "--scale",
        type=float,
        default=7.0,
        help="Steering condition scale: eps = eps(x, cond_steering) + scale * (eps(x, empty) - eps(x, cond_steering))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.1,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )

    parser.add_argument(
        "--opt_inversion",
        type=bool,
        default=False,
        help="Optimized inversion latent using latent search",
    )

    parser.add_argument(
        "--opt_inversion_itrs",
        type=int,
        default=20,
        help="number of iteration for inversion correction",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument(
        "--save_cond",
        type=bool,
        help="save the learned steering condition",
        default=True
    )

    parser.add_argument(
        "--cond_path",
        type=str,
        default=None,
        help="Path of the optimized steering condition",
    )


    opt = parser.parse_args()
    seed_everything(opt.seed)
    
    outpath = opt.outdir
    os.makedirs(outpath, exist_ok=True)
    shutil.copy(f"scripts/CondMapping_Aug.py", f"{outpath}/run_script.py")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f'{outpath}/output.log'),
            logging.StreamHandler()])
    
    logging.info(f"------------------ Script Arguments Start ----------------------")
    for k, v in vars(opt).items():
        logging.info(f"{k} {v}")
    logging.info(f"------------------ Script Arguments End ------------------------")

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)


    # assert os.path.isdir(opt.before_imgs)
    if opt.cond_path is None:

        ref_imgs = []
        input_imgs = []
        if os.path.isdir(opt.input_imgs):
            imgs_names = [f"{opt.input_imgs}/{fi}" for fi in os.listdir(opt.input_imgs)]
            imgs_names.sort()
            imgs_names = imgs_names[:opt.num_train_samples]
            ref_names = [f"{opt.reference_imgs}/{fi}" for fi in os.listdir(opt.reference_imgs)]
            ref_names.sort()
            ref_names = ref_names[:opt.num_train_samples]
        
        else:
            imgs_names = [opt.input_imgs]
            ref_names = [opt.ref_imgs]

        logging.info(f"Condition Optimization Examples: {ref_names}")

        for fi, ref_fi in zip(imgs_names, ref_names):
            input_imgs.append(Image.open(fi).convert("RGB"))
            ref_imgs.append(Image.open(ref_fi).convert("RGB"))

    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    logging.info(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    # with torch.no_grad():
    with precision_scope("cuda"):        
        with model.ema_scope():
            uc = None
            uc = model.get_learned_conditioning([""])
            tic = time.time()

            if opt.cond_path is None:

                c_out_new = cond_mapping_perstep_mlp(ref_imgs, input_imgs, t_enc, uc, sampler, opt.opt_cond_itrs, opt.scale, uc, batch_size=opt.batch_size)

                if opt.save_cond:
                    logging.info(f"saving learned condition with size {torch.cat(c_out_new).detach().cpu().numpy().shape}")
                    np.save(f"{outpath}/learned_conds.npy", torch.cat(c_out_new).detach().cpu().numpy())
            else:
                c_out_new= torch.from_numpy(np.load(opt.cond_path)).to(device).unsqueeze(1)

            # decode it
            with torch.no_grad():
                if os.path.isfile(opt.inference_imgs):
                    infer_names = [opt.inference_imgs]
                else:
                    infer_names = [f"{opt.inference_imgs}/{fi}" for fi in os.listdir(opt.inference_imgs)]
                    infer_names.sort()
                    if opt.num_inference_samples != 0 :
                        infer_names = infer_names[:opt.num_inference_samples]

                for infer_name in infer_names:
                    infer_image = load_img(infer_name).to(device)
                    infer_image = repeat(infer_image, '1 ... -> b ...', b=1)
                    infer_latent = model.get_first_stage_encoding(model.encode_first_stage(infer_image))  # move to latent space
                    z_inp =  sampler.ddim_inverse(infer_latent, uc, t_enc)[-1]
                    if opt.opt_inversion:
                        with torch.enable_grad():
                            z_inp = optimize_inversion(infer_image, infer_latent.detach(), z_inp.detach(), t_enc, uc, sampler, opt.opt_inversion_itrs, uc)
                    samples = sampler.decode_VDC(z_inp.detach(), uc, t_enc, unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc, perstep_cond=c_out_new)
                    
                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    if not opt.skip_save:
                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(outpath, infer_name.rsplit("/",1)[-1]))

    toc = time.time()

    logging.info(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
