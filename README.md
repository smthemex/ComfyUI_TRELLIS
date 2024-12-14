# ComfyUI_TRELLIS
<h3>You can use TRELLIS in comfyUI </h3>   

[TRELLIS](https://github.com/microsoft/TRELLIS/tree/main), Structured 3D Latents for Scalable and Versatile 3D Generation

---

# 1. Installation

In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_TRELLIS.git
```

---

# 2. Requirements  
本插件的测试环境是python3.11，torch2.5.1 cu124！！！    
The testing environment for this node is Python 3.11, torch2.5.1 cu124！！！   

```
pip install -r requirements.txt
```
以下必须要安装成功，否则无法运行!!!   
以下示例是按torch2.5.1 and cu124安装，你可以改成你当前环境的cu和torch，源于[issue3](https://github.com/microsoft/TRELLIS/issues/3)   
The following must be installed successfully, otherwise it cannot run !!!    
Example for torch2.5.1 and cu124,you can change to torch2.4.0  from [issue3](https://github.com/microsoft/TRELLIS/issues/3)   

xformers 和 flash-attention 可以只安装一项   
xformers and Flash Attention can be installed with only one option   

```
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
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
**Maybe Need**
* [kaolin](https://nvidia-kaolin.s3.us-east-2.amazonaws.com/index.html)   find  wheel，在这里找kaolin的各种版本轮子
* [flas attention](https://github.com/Dao-AILab/flash-attention/releases/)  find  wheel,在这里找flash attention的各种版本轮子






#  Example
-----
* opt模式目前有bug，贴图为全黑，fast模式正常使用，here is currently a bug in opt mode, the texture is all black, and fast mode works normally  
![](https://github.com/smthemex/ComfyUI_TRELLIS/blob/main/exmaple.png)
