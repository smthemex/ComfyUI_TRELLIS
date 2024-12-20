# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import uuid
import torch
from PIL import Image
import numpy as np
import cv2
import sys
import trimesh

from comfy.utils import common_upscale,ProgressBar
import folder_paths

#https://github.com/Abecid/realtime-3d-stylization-drawing/blob/97a687f148cec04e107f7fec8986c8b5615af1ec/gradio_app.py#L33
def glb2obj_(glb_path, obj_path):
    print('Converting glb to obj')
    mesh = trimesh.load(glb_path)
    
    if isinstance(mesh, trimesh.Scene):
        vertices = 0
        for g in mesh.geometry.values():
            vertices += g.vertices.shape[0]
    elif isinstance(mesh, trimesh.Trimesh):
        vertices = mesh.vertices.shape[0]
    else:
        raise ValueError('It is not mesh or scene')
    
    if vertices > 300000:
        raise ValueError('Too many vertices')
    if not os.path.exists(os.path.dirname(obj_path)):
        os.makedirs(os.path.dirname(obj_path))
    mesh.export(obj_path)
    print('Convert Done')

#https://github.com/robotflow-initiative/model_format_converter/blob/8f45efcfbec22444869548b369f7f77cdf9b04e4/model_format_converter/blender_scripts/obj2fbx.py#L6
def obj2fbx_(obj_path, fbx_path):
    # print all objects
    #for obj in bpy.data.objects:
    #    print(obj.name)
    import bpy
    if "Cube" in bpy.data.meshes:
        mesh = bpy.data.meshes["Cube"]
        print("removing mesh", mesh)
        bpy.data.meshes.remove(mesh)

    bpy.ops.import_scene.obj(filepath=obj_path)
    bpy.ops.export_scene.fbx(filepath=fbx_path)
    print('Convert Done')

def preprocess_image_(image: Image.Image,pipe,TMP_DIR):# -> Tuple[str, Image.Image]
    """
    Preprocess the input image.

    Args:
        image (Image.Image): The input image.

    Returns:
        str: uuid of the trial.
        Image.Image: The preprocessed image.
    """
    trial_id = str(uuid.uuid4())
    processed_image = pipe.preprocess_image(image)
    processed_image.save(f"{TMP_DIR}/{trial_id}.png")
    return trial_id, processed_image


def find_directories(base_path):
    directories = []
    for root, dirs, files in os.walk(base_path):
        for name in dirs:
            directories.append(name)
    return directories


def pil2narry(img):
    img = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return img


def narry_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        modified_value = pil2narry(value)
        list_in[i] = modified_value
    return list_in


def get_video_img(tensor):
    if tensor == None:
        return None
    outputs = []
    for x in tensor:
        x = tensor_to_pil(x)
        outputs.append(x)
    yield outputs

def instance_path(path, repo):
    if repo == "":
        if path == "none":
            repo = "none"
        else:
            model_path = get_local_path(folder_paths.base_path, path)
            repo = get_instance_path(model_path)
    return repo


def gen_img_form_video(tensor):
    pil = []
    for x in tensor:
        pil[x] = tensor_to_pil(x)
    yield pil


def phi_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        list_in[i] = value
    return list_in

def tensor_to_pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def nomarl_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor_to_pil(samples)
    return img_pil

def tensor_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    return samples

def get_local_path(comfy_file_path, model_path):
    path = os.path.join(comfy_file_path, "models", "diffusers", model_path)
    model_path = os.path.normpath(path)
    if sys.platform == 'win32':
        model_path = model_path.replace('\\', "/")
    return model_path

def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path


def tensor2cv(tensor_image):
    if len(tensor_image.shape)==4:# b hwc to hwc
        tensor_image=tensor_image.squeeze(0)
    if tensor_image.is_cuda:
        tensor_image = tensor_image.cpu()
    tensor_image=tensor_image.numpy()
    #反归一化
    maxValue=tensor_image.max()
    tensor_image=tensor_image*255/maxValue
    img_cv2=np.uint8(tensor_image)#32 to uint8
    img_cv2=cv2.cvtColor(img_cv2,cv2.COLOR_RGB2BGR)
    return img_cv2

def cvargb2tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def cv2tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img)
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256


def images_generator(img_list: list,):
    #get img size
    sizes = {}
    for image_ in img_list:
        if isinstance(image_,Image.Image):
            count = sizes.get(image_.size, 0)
            sizes[image_.size] = count + 1
        elif isinstance(image_,np.ndarray):
            count = sizes.get(image_.shape[:2][::-1], 0)
            sizes[image_.shape[:2][::-1]] = count + 1
        else:
            raise "unsupport image list,must be pil or cv2!!!"
    size = max(sizes.items(), key=lambda x: x[1])[0]
    yield size[0], size[1]
    
    # any to tensor
    def load_image(img_in):
        if isinstance(img_in, Image.Image):
            img_in=img_in.convert("RGB")
            i = np.array(img_in, dtype=np.float32)
            i = torch.from_numpy(i).div_(255)
            if i.shape[0] != size[1] or i.shape[1] != size[0]:
                i = torch.from_numpy(i).movedim(-1, 0).unsqueeze(0)
                i = common_upscale(i, size[0], size[1], "lanczos", "center")
                i = i.squeeze(0).movedim(0, -1).numpy()
            return i
        elif isinstance(img_in,np.ndarray):
            i=cv2.cvtColor(img_in,cv2.COLOR_BGR2RGB).astype(np.float32)
            i = torch.from_numpy(i).div_(255)
            #print(i.shape)
            return i
        else:
           raise "unsupport image list,must be pil,cv2 or tensor!!!"
        
    total_images = len(img_list)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(load_image, img_list)
    try:
        prev_image = next(images)
        while True:
            next_image = next(images)
            yield prev_image
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image = next_image
    except StopIteration:
        pass
    if prev_image is not None:
        yield prev_image

def load_images(img_list: list,):
    gen = images_generator(img_list)
    (width, height) = next(gen)
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded .")
    return images

def tensor2pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def tensor2imglist(image):# pil
    B, _, _, _ = image.size()
    if B == 1:
        list_out = [tensor2pil_preprocess(image)]
    else:
        image_list = torch.chunk(image, chunks=B)
        list_out = [tensor2pil_preprocess(i) for i in image_list]
    return list_out,B

def tensor2pil_preprocess(image):
    cv_image = tensor2cv(image)
    cv_image = center_resize_pad(cv_image, 512, 512)
    img=cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    iamge_pil=Image.fromarray(img)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    iamge_pil.save(f"{timestamp}_b.png")
    return  iamge_pil

def cf_tensor2cv(tensor,width, height):
    d1, _, _, _ = tensor.size()
    if d1 > 1:
        tensor_list = list(torch.chunk(tensor, chunks=d1))
        tensor = [tensor_list][0]
    cr_tensor=tensor_upscale(tensor,width, height)
    cv_img=tensor2cv(cr_tensor)
    return cv_img

def pre_img(tensor,max):
    cv_image = tensor2cv(tensor) #转CV
    h, w = cv_image.shape[:2]
    cv_image = center_resize_pad(cv_image, h, h) #以高度中心裁切或填充
    cv_image=cv2.resize(cv_image, (max, max)) #缩放到统一高度
    return  cv2tensor(cv_image)

def center_resize_pad(img, new_width, new_height):#模型尺寸推荐518
    h, w = img.shape[:2]
    if w == h:
        if w == new_width:
            return img
        else:
            return cv2.resize(img, (new_width, new_height))
    else: #蒙版也有可能不是正方形
        if h > w:  # 竖直图左右填充
            s = max(h, w)
            f = np.zeros((s, s, 3), np.uint8)
            ax, ay = (s - img.shape[1]) // 2, (s - img.shape[0]) // 2
            f[ay:img.shape[0] + ay, ax:ax + img.shape[1]] = img
        else:
            f = center_crop(img, h, h)
        return cv2.resize(f, (new_width, new_height))


def center_crop(image, crop_width, crop_height):
    # 获取图像的中心坐标
    height, width = image.shape[:2]
    x = width // 2 - crop_width // 2
    y = height // 2 - crop_height // 2
    
    x=max(0,x)
    y=max(0,y)
    
    # 裁剪图像
    crop_img = image[y:y + crop_height, x:x + crop_width]
    return crop_img
