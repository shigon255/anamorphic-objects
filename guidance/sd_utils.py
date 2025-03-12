from diffusers import (
    DDIMScheduler,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusionPipeline,
    StableDiffusion3Pipeline,
)
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.import_utils import is_xformers_available


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        sd_version="2.1",
        hf_key=None,
        t_range=[0.02, 0.98],
        sds_bridge=False,
        cache_dir=None,
    ):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "2.1":
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == "2.0":
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == "1.5":
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )

        self.dtype = torch.float16 if fp16 else torch.float32

        # Create model
        if cache_dir is not None:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_key, torch_dtype=self.dtype, cache_dir=cache_dir
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_key, torch_dtype=self.dtype
            )

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        
        if cache_dir is not None:
            self.scheduler = DDIMScheduler.from_pretrained(
                model_key, subfolder="scheduler", torch_dtype=self.dtype, cache_dir=cache_dir
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                model_key, subfolder="scheduler", torch_dtype=self.dtype
            )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = {}
        
        self.sds_bridge = sds_bridge
        if sds_bridge:
            self.src_prompt = "oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed"
            self.embeddings['src'] = self.encode_text([self.src_prompt])

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds

        # directional embeddings
        for d in ['front', 'side', 'back']:
            embeds = self.encode_text([f'{p}, {d} view' for p in prompts])
            self.embeddings[d] = embeds
    
    @torch.no_grad()
    def get_text_embeds_multiview(self, prompts, negative_prompts):
        basic_prompt = prompts['basic']
        pos_embeds = self.encode_text(basic_prompt)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds

        # directional embeddings
        for d in ['front', 'left', 'right', 'back', 'left_front', 'right_front', 'left_back', 'right_back']:
            # embeds = self.encode_text([f'{prompts[d]}, {d} view'])
            embeds = self.encode_text([f'{prompts[d]}'])
            self.embeddings[d] = embeds
    
    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    @torch.no_grad()
    def refine(self, pred_rgb,
               guidance_scale=100, steps=50, strength=0.8,
        ):

        batch_size = pred_rgb.shape[0]
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512.to(self.dtype))
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])
        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
    
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=embeddings,
            ).sample

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        return imgs

    def train_step(
        self,
        pred_rgb,
        step_ratio=None,
        guidance_scale=100,
        as_latent=False,
        vers=None, hors=None,
    ):
        
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        with torch.no_grad():
            if step_ratio is not None:
                # dreamtime-like
                # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            else:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

            # predict the noise residual with unet, NO grad!
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            if hors is None:
                if self.sds_bridge:
                    embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['src'].expand(batch_size, -1, -1)])
                else:
                    embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])
            else:
                def _get_dir_ind(h):
                    if abs(h) < 30: return 'front'
                    elif abs(h) > 150: return 'back'
                    elif 30 <= h < 60: return 'right_front'
                    elif 60 <= h < 120: return 'right'
                    elif 120 <= h < 150: return 'right_back'
                    elif -30 > h > -60: return 'left_front'
                    elif -60 > h > -120: return 'left'
                    elif -120 > h > -150: return 'left_back'
                    else: return 'back'
                # import pdb; pdb.set_trace()
                if self.sds_bridge:
                    embeddings = torch.cat([self.embeddings[_get_dir_ind(h)] for h in hors] + [self.embeddings['src'].expand(batch_size, -1, -1)])
                else:
                    embeddings = torch.cat([self.embeddings[_get_dir_ind(h)] for h in hors] + [self.embeddings['neg'].expand(batch_size, -1, -1)])

            noise_pred = self.unet(
                latent_model_input, tt, encoder_hidden_states=embeddings
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)

            # seems important to avoid NaN...
            # grad = grad.clamp(-1, 1)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]

        return loss

    @torch.no_grad()
    def produce_latents(
        self,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    1,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        batch_size = latents.shape[0]
        self.scheduler.set_timesteps(num_inference_steps)
        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=embeddings
            ).sample

            # perform guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)
        
        # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs

class StableDiffusion3(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        sd_version="3.0",
        hf_key=None,
        t_range=[0.02, 0.98],
        cache_dir=None,
    ):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "3.0":
            model_key = "stabilityai/stable-diffusion-3-medium-diffusers"
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )

        self.dtype = torch.float16 if fp16 else torch.float32

        # Create model
        if cache_dir is not None:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_key, torch_dtype=self.dtype, cache_dir=cache_dir
            )
        else:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_key, torch_dtype=self.dtype
            )            
        
        # if vram_O:
        #     pipe.enable_sequential_cpu_offload()
        #     pipe.enable_vae_slicing()
        #     pipe.unet.to(memory_format=torch.channels_last)
        #     pipe.enable_attention_slicing(1)
        #     # pipe.enable_model_cpu_offload()
        # else:
        pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.text_encoder_3 = pipe.text_encoder_3
        self.transformer = pipe.transformer

        # self.scheduler = DDIMScheduler.from_pretrained(
        #     model_key, subfolder="scheduler", torch_dtype=self.dtype
        # )
        # self.scheduler = pipe.scheduler
        if cache_dir is not None:
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                model_key, subfolder="scheduler", torch_dtype=self.dtype, cache_dir=cache_dir
            )
        else:
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                model_key, subfolder="scheduler", torch_dtype=self.dtype
            )

        del pipe
        # self.pipe = pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        # self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = {}
        self.pooled_embeddings = {}

    # @torch.no_grad()
    # def get_text_embeds_pipe(self, prompts, negative_prompts):
    #     (
    #         prompt_embeds,
    #         negative_prompt_embeds,
    #         pooled_prompt_embeds,
    #         negative_pooled_prompt_embeds,
    #     ) = self.pipe.encode_prompt(
    #         prompt=prompts,
    #         prompt_2=prompts,
    #         prompt_3=prompts,
    #         negative_prompt=negative_prompts,
    #         negative_prompt_2=negative_prompts,
    #         negative_prompt_3=negative_prompts,
    #         # do_classifier_free_guidance=self.do_classifier_free_guidance,
    #         # prompt_embeds=prompt_embeds,
    #         # negative_prompt_embeds=negative_prompt_embeds,
    #         # pooled_prompt_embeds=pooled_prompt_embeds,
    #         # negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    #         device=device,
    #         # clip_skip=self.clip_skip,
    #         # num_images_per_prompt=num_images_per_prompt,
    #         # max_sequence_length=max_sequence_length,
    #         # lora_scale=lora_scale,
    #     )

    #     self.embeddings['pos'] = prompt_embeds
    #     self.embeddings['neg'] = negative_prompt_embeds
    #     self.pooled_embeddings['pos'] = pooled_prompt_embeds
    #     self.pooled_embeddings['neg'] = negative_pooled_prompt_embeds
        

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds, pos_pooled_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds, neg_pooled_embeds = self.encode_text(negative_prompts)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds
        self.pooled_embeddings['pos'] = pos_pooled_embeds
        self.pooled_embeddings['neg'] = neg_pooled_embeds

        # directional embeddings
        for d in ['front', 'side', 'back']:
            embeds, pooled_embeds = self.encode_text([f'{p}, {d} view' for p in prompts])
            self.embeddings[d] = embeds
            self.pooled_embeddings[d] = pooled_embeds
    
    @torch.no_grad()
    def get_text_embeds_multiview(self, prompts, negative_prompts):
        basic_prompt = prompts['basic']
        pos_embeds, pos_pooled_embeds = self.encode_text(basic_prompt)  # [1, 77, 768]
        neg_embeds, neg_pooled_embeds = self.encode_text(negative_prompts)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds
        self.pooled_embeddings['pos'] = pos_pooled_embeds
        self.pooled_embeddings['neg'] = neg_pooled_embeds

        # directional embeddings
        for d in ['front', 'side', 'back']:
            embeds, pooled_embeds = self.encode_text([f'{prompts[d]}, {d} view'])
            self.embeddings[d] = embeds
            self.pooled_embeddings[d] = pooled_embeds
    
    def encode_text(self, prompt):
        # prompt: [str]
        embeddings1, pooled_embeddings1 = self.encode_text_clip(prompt, clip_model_index=0)
        embeddings2, pooled_embeddings2 = self.encode_text_clip(prompt, clip_model_index=1)
        pooled_embeddings = torch.cat([pooled_embeddings1, pooled_embeddings2], dim=-1)
        clip_embeddings = torch.cat([embeddings1, embeddings2], dim=-1)
        t5_embeddings = self.encode_text_t5(prompt)
        clip_embeddings = torch.nn.functional.pad(
            clip_embeddings, (0, t5_embeddings.shape[-1] - clip_embeddings.shape[-1])
        )

        # import pdb; pdb.set_trace()
        embeddings = torch.cat([clip_embeddings, t5_embeddings], dim=-2)
        return embeddings, pooled_embeddings    

    def encode_text_clip(self, prompt, clip_model_index=0):
        # prompt: [str]
        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            # truncation=True,
            return_tensors="pt",
        )
        embeddings = text_encoder(inputs.input_ids.to(self.device), output_hidden_states=True)
        pooled_embebeddings = embeddings[0]
        embeddings = embeddings.hidden_states[-2]
        return embeddings, pooled_embebeddings

    def encode_text_t5(self, prompt):
        inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            # truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        embeddings = self.text_encoder_3(inputs.input_ids.to(self.device))[0]
        return embeddings

    def add_noise(self, latents, t, noise):
        sigma = (t-self.min_step) / (self.max_step-self.min_step)
        latents_noisy = sigma * noise + (1.0 - sigma) * latents
        latents_noisy = latents_noisy.to(latents.dtype)
        return latents_noisy

    def train_step(
        self,
        pred_rgb,
        step_ratio=None,
        guidance_scale=100,
        as_latent=False,
        vers=None, hors=None,
    ):
        
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        with torch.no_grad():
            if step_ratio is not None:
                # dreamtime-like
                # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            else:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

            # (For SD2)
            #   w(t), sigma_t^2
            #    w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)
            # (For SD3)
            #   w = (t/(1-t))(2/(pi-2pit+2pit^2))
            sigma = (t-self.min_step) / (self.max_step-self.min_step)
            # w = (sigma/(1-sigma))*(2/(np.pi-2*np.pi*sigma+2*np.pi*sigma**2)).view(batch_size, 1, 1, 1)
            w = torch.ones(batch_size, 1, 1, 1, device=self.device)

            # predict the noise residual with unet, NO grad!
            # add noise
            noise = torch.randn_like(latents)
            # latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # latents_noisy = self.scheduler.scale_noise(latents, t, noise)
            latents_noisy = self.add_noise(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            if hors is None:
                embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])
                pooled_embeddings = torch.cat([self.pooled_embeddings['pos'].expand(batch_size, -1), self.pooled_embeddings['neg'].expand(batch_size, -1)])            
            else:
                def _get_dir_ind(h):
                    if abs(h) < 60: return 'front'
                    elif abs(h) < 120: return 'side'
                    else: return 'back'
                embeddings = torch.cat([self.embeddings[_get_dir_ind(h)] for h in hors] + [self.embeddings['neg'].expand(batch_size, -1, -1)])
                pooled_embeddings = torch.cat([self.pooled_embeddings[_get_dir_ind(h)] for h in hors] + [self.pooled_embeddings['neg'].expand(batch_size, -1)])

            # noise_pred = self.unet(
            #     latent_model_input, tt, encoder_hidden_states=embeddings
            # ).sample

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=tt,
                encoder_hidden_states=embeddings,
                pooled_projections=pooled_embeddings,
                # joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            # import pdb; pdb.set_trace()
            # perform guidance (high scale from paper!)
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            grad = w * (noise_pred - noise)
            # import pdb; pdb.set_trace()
            grad = torch.nan_to_num(grad)
            # seems important to avoid NaN...
            # grad = grad.clamp(-1, 1)
      

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]

        return loss

    @torch.no_grad()
    def produce_latents(
        self,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    1,
                    16, # hardcoded for now
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        batch_size = latents.shape[0]
        self.scheduler.set_timesteps(num_inference_steps)
        embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])
        pooled_embeddings = torch.cat([self.pooled_embeddings['pos'].expand(batch_size, -1), self.pooled_embeddings['neg'].expand(batch_size, -1)])
        
        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            device = latent_model_input.device
            timestep = t.expand(latent_model_input.shape[0]).to(device)
            # predict the noise residual
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=embeddings,
                pooled_projections=pooled_embeddings,
                # joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)[0]

        return latents

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = (posterior.sample() - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents \
            + self.vae.config.shift_factor

        image = self.vae.decode(latents, return_dict=False)[0]
        
        imgs = (image / 2 + 0.5).clamp(0, 1)
        # imgs = self.pipe.image_processor.postprocess(image)

        return imgs

    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=28,
        guidance_scale=7.0,
        latents=None,
    ):
        # import pdb; pdb.set_trace()
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)
        # self.get_text_embeds_pipe(prompts, negative_prompts)
        
        # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument(
        "--sd_version",
        type=str,
        default="2.1",
        choices=["1.5", "2.0", "2.1"],
        help="stable diffusion version",
    )
    parser.add_argument(
        "--hf_key",
        type=str,
        default=None,
        help="hugging face Stable diffusion model key",
    )
    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument(
        "--vram_O", action="store_true", help="optimization for low VRAM usage"
    )
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda")

    # sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)
    sd3 = StableDiffusion3(device, opt.fp16, opt.vram_O, "3.0", opt.hf_key)

    # imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)
    imgs = sd3.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)
    # imgs = sd3.pipe(
    #     "A cat holding a sign that says hello world",
    #     negative_prompt="",
    #     num_inference_steps=28,
    #     guidance_scale=7.0,
    # ).images

    # visualize image
    plt.imshow(imgs[0])
    # plt.show()
    plt.savefig("output.png")

