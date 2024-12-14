import os.path

import numpy as np
from .trellis.utils import render_utils, postprocessing_utils
import imageio
import folder_paths

MAX_SEED = np.iinfo(np.int32).max

def image_to_3d(pipeline,image,preprocess_image:bool,covert2video:bool,trial_id: str, seed: int,ss_guidance_strength: float, ss_sampling_steps: int, slat_guidance_strength: float, slat_sampling_steps: int,mesh_simplify,texture_size,mode):
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
    if covert2video:
        video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
        video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
        video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
        video_path = f"{trial_id}.mp4"
        imageio.mimsave(video_path, video, fps=15)
    
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



