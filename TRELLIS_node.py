# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import gc
import os
import torch
import uuid
from .app import image_to_3d
from .trellis.pipelines import TrellisImageTo3DPipeline
from .utils import tensor_to_pil,glb2obj,obj2fbx
import folder_paths

MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
current_path = os.path.dirname(os.path.abspath(__file__))

weigths_dinov2_current_path = os.path.join(folder_paths.models_dir, "dinov2")
if not os.path.exists(weigths_dinov2_current_path):
    os.makedirs(weigths_dinov2_current_path)

try:
    folder_paths.add_model_folder_path("dinov2", weigths_dinov2_current_path, False)
except:
    folder_paths.add_model_folder_path("dinov2", weigths_dinov2_current_path)

class Trellis_LoadModel:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo": ("STRING", {"default": "F:/test/ComfyUI/models/diffusers/JeffreyXiang/TRELLIS-image-large"}),
                "dino": (["none"] + folder_paths.get_filename_list("dinov2"),),
                "attn_backend":(["xformers","flash-attn"],),
                "spconv_algo":(["auto","flash-native"],),
            }
        }
    
    RETURN_TYPES = ("MODEL_TRELLIS", )
    RETURN_NAMES = ("model",)
    FUNCTION = "main_loader"
    CATEGORY = "Trellis"
    
    def main_loader(self, repo,dino,attn_backend,spconv_algo):
        if attn_backend=="xformers":
            os.environ['ATTN_BACKEND'] = 'xformers'
        else:
            os.environ['ATTN_BACKEND'] = 'flash-attn'
        if spconv_algo=="auto":
            os.environ['SPCONV_ALGO'] = 'auto'
        else:
            os.environ['SPCONV_ALGO'] = 'native'
        
        if dino=="none":
            raise "need choice dinov2 checkpoint"
        
        TrellisImageTo3DPipeline.dino=folder_paths.get_full_path("dinov2", dino)
        TrellisImageTo3DPipeline.dino_moudel=os.path.join(current_path,"facebookresearch/dinov2")
        if repo:
            model=TrellisImageTo3DPipeline.from_pretrained(repo)
        else:
            raise "need fill repo"
        return (model,)


class Trellis_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        
        return {
            "required": {
                "image": ("IMAGE",),  # [B,H,W,C], C=3
                "model": ("MODEL_TRELLIS",),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "slat_cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "slat_steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "preprocess_image": ("BOOLEAN", {"default": False},),
                "texture_size": ("INT", {"default": 512, "min": 512, "max": 2048, "step": 512, "display": "number"}),
                "mesh_simplify": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 0.98, "step": 0.01}),
                "mode":(["fast","opt"],),
                "covert2video": ("BOOLEAN", {"default": False},),
                "glb2obj": ("BOOLEAN", {"default": False},),
                "glb2fbx": ("BOOLEAN", {"default": False},),
            }
        }
    
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "sampler_main"
    CATEGORY = "Trellis"
    
    def sampler_main(self, image, model,  seed, cfg, steps,slat_cfg, slat_steps,preprocess_image,texture_size,mesh_simplify,mode,covert2video,glb2obj,glb2fbx):

        model.cuda()
        image = tensor_to_pil(image) #pil
        trial_id = str(uuid.uuid4())
        glb=image_to_3d(model,image,preprocess_image,covert2video,trial_id,seed,cfg,steps,slat_cfg,slat_steps,mesh_simplify,texture_size,mode)
        glb_path = f"{folder_paths.get_output_directory()}/{trial_id}.glb"
        glb.export(glb_path)
        print(f"glb save in {glb_path} ")
        #model.cpu()
        #gc.collect()
        torch.cuda.empty_cache()
        if glb2obj:
            obj_path=f"{folder_paths.get_output_directory()}/{trial_id}.obj"
            glb2obj(glb_path, obj_path)
            glb_path=obj_path
        if glb2fbx:
            obj_path = f"{folder_paths.get_output_directory()}/{trial_id}.obj"
            glb2obj(glb_path, obj_path)
            fbx_path = f"{folder_paths.get_output_directory()}/{trial_id}.fbx"
            obj2fbx(obj_path,fbx_path)
            glb_path = fbx_path
        return (glb_path,)


NODE_CLASS_MAPPINGS = {
    "Trellis_LoadModel": Trellis_LoadModel,
    "Trellis_Sampler": Trellis_Sampler,

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis_LoadModel": "Trellis_LoadModel",
    "Trellis_Sampler": "Trellis_Sampler",

}
