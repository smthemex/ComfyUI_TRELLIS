# ComfyUI_TRELLIS
<h3>You can use TRELLIS in comfyUI </h3>   

[TRELLIS](https://github.com/microsoft/TRELLIS/tree/main), Structured 3D Latents for Scalable and Versatile 3D Generation

---

# Notice 注意
* 只支持安装版的comfyUI，便携包目前测试无法运行（想了不少办法都不行），如果你用的是便携包，请暂时不要安装，秋叶包我没有，所以不知道能不能用。
* Only supports the installation version of ComfyUI. The portable package is currently unable to run during testing (despite trying various methods). If you are using the portable package, please do not install it temporarily

  
# 1. Installation

In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_TRELLIS.git
```

---

# 2. Requirements  
本插件的测试环境是python3.11，torch2.5.1 cu124...       
The testing environment for this node is Python 3.11, torch2.5.1 cu124...    

```
pip install -r requirements.txt
```
以下必须要安装成功，否则无法运行!!!   
以下示例是按torch2.5.1 and cu124安装，你可以改成你当前环境的cu和torch，源于[issue3](https://github.com/microsoft/TRELLIS/issues/3)   

The following must be installed successfully, otherwise it cannot run !!!    
Example for torch2.5.1 and cu124,you can change to torch2.4.0 or other  from [issue3](https://github.com/microsoft/TRELLIS/issues/3)   

xformers 和 flash-attention 可以只安装一项   
xformers and Flash Attention can be installed with only one option   

```
pip install https://github.com/bdashore3/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu124torch2.5.1cxx11abiFALSE-cp310-cp310-win_amd64.whl
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html

git clone https://github.com/NVlabs/nvdiffrast.git ./tmp/extensions/nvdiffrast
pip install ./tmp/extensions/nvdiffrast

git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git ./tmp/extensions/diffoctreerast
pip install ./tmp/extensions/diffoctreerast

git clone https://github.com/autonomousvision/mip-splatting.git ./tmp/extensions/mip-splatting
pip install ./tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

# 在..ComfyUI_TRELLIS目录下，复制vox2seq插件目录到temp，然后pip安装 Under ComfyUI_TRELLIS directory，Copy the Vox2seq plugin to temp and install it using pip  
cp -r ./extensions/vox2seq ./tmp/extensions/vox2seq
pip install ./tmp/extensions/vox2seq

pip install spconv-cu120  #这个最后安装 This is the final installation

```


**2.1Other Need**
* [kaolin](https://nvidia-kaolin.s3.us-east-2.amazonaws.com/index.html)   find  wheel，在这里找kaolin的各种版本轮子
* [flas attention](https://github.com/Dao-AILab/flash-attention/releases/)  find  wheel,在这里找flash attention的各种版本轮子
* [visualstudio](https://visualstudio.microsoft.com/zh-hans/)   visual studio2019 or high   windows必须安装

**2.2visualstudio & cuda**
* 必须将visualstudio的cl.exe加入系统的环境变量path中，以下是windows系统示例，具体以自己的系统目录为准; 
* The cl.exe of VisualStudio must be added to the system's environment variable path. Here is an example, please refer to your own system directory for details;  
```
 Path:        C:\Program Files(x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.28610\bin\Hostx64\x64 # or  other version 或者其他版本
 Path:        C:\Program Files(x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.28610\bin\Hostx64\x64\cl.exe # or  other version 或者其他版本
 Path:        C:\Users\yourname\AppData\Roaming\Python\Python311\Scripts # python 
 CUDA_PATH:   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4 # or  other version 或者其他版本

```


# 3. Models Required 
* 3.1 TRELLIS_repo,download or using online...  
[JeffreyXiang/TRELLIS-image-large](https://huggingface.co/JeffreyXiang/TRELLIS-image-large)   
如果预下载下 ，在repo位置填写：x:/your/path/JeffreyXiang/TRELLIS-image-large  
if pre download ,fill local path in repo like this: x:/your/path/JeffreyXiang/TRELLIS-image-large

```
├── anypath/JeffreyXiang/TRELLIS-image-large/
|   ├── pipeline.json
|   ├── ckpts/
|            ├── slat_dec_gs_swin8_B_64l8gs32_fp16.json
|            ├── slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors
|            ├── slat_dec_mesh_swin8_B_64l8m256c_fp16.json
|            ├── slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors
|            ├── slat_dec_rf_swin8_B_64l8r16_fp16.json
|            ├── slat_dec_rf_swin8_B_64l8r16_fp16.safetensors
|            ├── slat_enc_swin8_B_64l8_fp16.json
|            ├── slat_enc_swin8_B_64l8_fp16.safetensors
|            ├── slat_flow_img_dit_L_64l8p2_fp16.json
|            ├── slat_flow_img_dit_L_64l8p2_fp16.safetensors
|            ├── ss_dec_conv3d_16l8_fp16.json
|            ├── ss_dec_conv3d_16l8_fp16.safetensors
|            ├── ss_enc_conv3d_16l8_fp16.json
|            ├── ss_enc_conv3d_16l8_fp16.safetensors
|            ├── ss_flow_img_dit_L_16l8_fp16.json
|            ├── ss_flow_img_dit_L_16l8_fp16.safetensors
```
* 3.2 dinov2  
因为官方的代码每次加载dinov2都要连GitHub，所以我改成了离线版的，需要下载dinov2模型，[地址](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth) 
模型放在comfyUI/models/dinov2目录下
```
├── ComfyUI/models/dinov2
|      ├── dinov2_vitl14_reg4_pretrain.pth
```

---

# 4 Example

* opt模式目前有bug，贴图为全黑，fast模式正常使用，here is currently a bug in opt mode, the texture is all black, and fast mode works normally  
![](https://github.com/smthemex/ComfyUI_TRELLIS/blob/main/exmaple.png)


---


# 5 Citation

microsoft/TRELLIS
```
@article{xiang2024structured,
    title   = {Structured 3D Latents for Scalable and Versatile 3D Generation},
    author  = {Xiang, Jianfeng and Lv, Zelong and Xu, Sicheng and Deng, Yu and Wang, Ruicheng and Zhang, Bowen and Chen, Dong and Tong, Xin and Yang, Jiaolong},
    journal = {arXiv preprint arXiv:2412.01506},
    year    = {2024}
}
```
facebookresearch/dinov2
```
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```
```
@misc{darcet2023vitneedreg,
  title={Vision Transformers Need Registers},
  author={Darcet, Timothée and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  journal={arXiv:2309.16588},
  year={2023}
}
```

