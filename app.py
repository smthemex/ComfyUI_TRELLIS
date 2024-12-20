import os.path

import numpy as np
from .trellis.utils import render_utils, postprocessing_utils
import imageio
import folder_paths

MAX_SEED = np.iinfo(np.int32).max

def image_to_3d(pipeline,image,preprocess_image:bool,covert2video:bool,trial_id: str, seed: int,ss_guidance_strength: float, ss_sampling_steps: int, slat_guidance_strength: float, slat_sampling_steps: int,mesh_simplify,texture_size,mode,is_multiimage,Gaussians2PLY,multiimage_algo):
    """
    Convert an image to a 3D model.

    Args:
        trial_id (str): The uuid of the trial.
        seed (int): The random seed.
        randomize_seed (bool): Whether to randomize the seed.
        ss_guidance_strength (float): The guidance strength for sparse structure generation.
        ss_sampling_steps (int): The number of sampling steps for sparse structure generation.
        slat_guidance_strength (float): The guidance strength for structured latent generation.
        slat_sampling_steps (int): The number of sampling steps for structured latent generation.

    Returns:
    """
    # if randomize_seed:
    #     seed = np.random.randint(0, MAX_SEED)
    if not  is_multiimage:
        outputs = pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=preprocess_image,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
    else:
        outputs = pipeline.run_multi_image(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=preprocess_image,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
            mode=multiimage_algo,
        )
      
    if covert2video:
        video_path = f"{trial_id}.mp4"
        if is_multiimage:
            video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
            video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
            video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in
                     zip(video_gs, video_mesh)]
            imageio.mimsave(video_path, video, fps=30)
        else:
            video = render_utils.render_video(outputs['gaussian'][0])['color']
            imageio.mimsave(f"{trial_id}_gs.mp4", video, fps=30)
            video = render_utils.render_video(outputs['radiance_field'][0])['color']
            imageio.mimsave(f"{trial_id}_rf.mp4", video, fps=30)
            video = render_utils.render_video(outputs['mesh'][0])['normal']
            imageio.mimsave(f"{trial_id}_mesh.mp4", video, fps=30)
       
    if Gaussians2PLY:
        outputs['gaussian'][0].save_ply(f"{trial_id}.ply")
    
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=mesh_simplify,  # Ratio of triangles to remove in the simplification process
        texture_size=texture_size,  # Size of the texture used for the GLB
        mode=mode,
        uv_map=f"{folder_paths.get_output_directory()}/{trial_id}_uv_map.png",
    )
    
    return glb



