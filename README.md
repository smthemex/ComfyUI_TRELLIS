# ComfyUI_TRELLIS
<h3>You can use TRELLIS in comfyUI </h3>   

[TRELLIS](https://github.com/microsoft/TRELLIS/tree/main), Structured 3D Latents for Scalable and Versatile 3D Generation

---

# Update  
* 2025/03/27 kaolin support torch2.6 now;    

  
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
# if torch 2.6
git clone git@github.com:NVIDIAGameWorks/kaolin.git
cd kaolin
pip install .

git clone https://github.com/NVlabs/nvdiffrast.git ./tmp/extensions/nvdiffrast
pip install ./tmp/extensions/nvdiffrast
#if install nvdiffrast error ，see below how to fix it 

git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git ./tmp/extensions/diffoctreerast
pip install ./tmp/extensions/diffoctreerast

git clone https://github.com/autonomousvision/mip-splatting.git ./tmp/extensions/mip-splatting
pip install ./tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/
# if update torch2.6 ,must del diff-gaussian-rasterization and reinstall it

pip install spconv-cu120	 #if cuda>120
# pip install spconv-cu118  # if cuda118 

# 在..ComfyUI_TRELLIS目录下，复制vox2seq插件目录到temp，然后pip安装 Under ComfyUI_TRELLIS directory，Copy the Vox2seq plugin to temp and install it using pip  
cp -r ./extensions/vox2seq ./tmp/extensions/vox2seq
pip install ./tmp/extensions/vox2seq

```

**2.1 Other Need**
* [kaolin](https://nvidia-kaolin.s3.us-east-2.amazonaws.com/index.html)   find  wheel，or install [normal](https://github.com/NVIDIAGameWorks/kaolin)在这里找kaolin的各种版本轮子
* [flash attention](https://github.com/Dao-AILab/flash-attention/releases/)  find  wheel,在这里找flash attention的各种版本轮子
* [visualstudio](https://visualstudio.microsoft.com/zh-hans/)   visual studio2019 or high   windows必须安装
* [spconv](https://github.com/traveller59/spconv)  find your cuda version ,if version.120 use spconv-cu120  cuda版本大于120的只能用spconv-cu120，其他根据对应地址版本安装
* if somebody install nvdiffrast fail can see how to  fix it  in [here  ](https://www.bilibili.com/video/BV1PMkEYzE8h/?vd_source=602446aa977e356a8a57180ba0877271)
* if install utils3d fail can see how to fix it in [here](https://github.com/smthemex/ComfyUI_TRELLIS/issues/6) @planb788

**2.2 visualstudio & cuda**
* 必须将visualstudio的cl.exe加入系统的环境变量path中，以下是windows系统示例，具体以自己的系统目录为准; 
* The cl.exe of VisualStudio must be added to the system's environment variable path. Here is an example, please refer to your own system directory for details;  
```
 Path:        C:\Program Files(x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.28610\bin\Hostx64\x64 # or  other version 或者其他版本
 Path:        C:\Program Files(x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.28610\bin\Hostx64\x64\cl.exe # or  other version 或者其他版本
 Path:        C:\Users\yourname\AppData\Roaming\Python\Python311\Scripts # python 
 CUDA_PATH:   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4 # or  other version 或者其他版本

```

**2.3 if use glb2fbx**   
Need ' pip install bpy ' and install ' blender ' 


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

![](https://github.com/smthemex/ComfyUI_TRELLIS/blob/main/example1220.png)
![](https://github.com/smthemex/ComfyUI_TRELLIS/blob/main/exmaple.png)
![](https://github.com/smthemex/ComfyUI_TRELLIS/blob/main/batch_example.png
)


**previous update**
* 根据TRELLIS的更新，新增三视参考图渲染模式（示例图为实际效果，此模式速度更快）和高斯保存按钮，针对三视参考图新增图片加载节点（内置正方形裁切),三视图模式开启multi_image生效，否则是常规的单图模式（会输出三个结果）；
* 如果你输入的图片不是纯色背景，建议开启preprocess_image以获得最好的效果。（此次更新也修复了加载RGBA图片可能导致的变形错误）；
*  使用 [here](https://github.com/smthemex/ComfyUI_TRELLIS/issues/6) @planb788 方法，我制作了python3.11，torch2.5.1 cu124.的便携包，可以在[Google dirver](https://drive.google.com/file/d/174StpwP3D1qSD0RuhM1XwTene-5VMiL4/view?usp=drive_link)  或者 [夸克网盘](https://pan.quark.cn/s/8e07717bdff7)下载，注意，即便是便携包，也是需要配置VS和python的系统变量路径的；
* 增加批量渲染功能，注意过多图片可能会OOM；  
* According to TRELLIS' update, a three view reference image rendering mode (the example image is the actual effect, which is faster) and a Gaussian save button have been added. For the three view reference image, a new image loading node (with built-in square cropping) has been added,Enabling multi_image in three view mode takes effect, otherwise it will be in regular single image mode (outputting three results);
* If the image you input is not a solid color background, it is recommended to enable 'preprocess_image' for the best effect. (This update also fixes deformation errors that may occur when loading RGBA images);
*  Use [here](https://github.com/smthemex/ComfyUI_TRELLIS/issues/6) , I have created a portable package for ‘Python 3.11, Torch 2.5.1, and CU124’ using the @planb788 method, which can be found on [Google dirver](https://drive.google.com/file/d/174StpwP3D1qSD0RuhM1XwTene-5VMiL4/view?usp=drive_link) Or [夸克网盘](https://pan.quark.cn/s/8e07717bdff7) Download, note that even portable packages require configuring the system variable paths for VS and Python;
* Add batch rendering function, be aware that too many images may cause OOM    

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

