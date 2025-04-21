# video_llama_XPU

1. Clone the Repository
Start by cloning the Video-LLaMA repository from GitHub:
bash
git clone https://github.com/DAMO-NLP-SG/Video-LLaMA.git
2. Install Dependencies
Navigate into the cloned directory and install the required Python packages:
bash
cd Video-LLaMA
pip3 install -r requirements.txt
Note: Ensure that the file is named requirements.txt (not requirement.txt), as the latter would cause an error during installation.
________________________________________
⚠️ Common Issues and Fixes

Issue 1: ModuleNotFoundError: No module named 'torchaudio'
![image](https://github.com/user-attachments/assets/0ade8b16-c3f0-4e1a-abb7-6e367661c636)

Cause: The torchaudio library is not installed. 
 
Solution:
```
pip install torchaudio
```
________________________________________
Issue 2: ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
![image](https://github.com/user-attachments/assets/fecfb113-e69f-498d-b782-34700b12aee7)

Cause: This is a known issue with the torchvision library. 
 
Solution:
1.	Navigate to the augmentations.py file:
bash
cd /lib/python3.x/site-packages/pytorchvideo/transforms/
2.	Open augmentations.py and comment out the problematic import statement:
### import torchvision.transforms.functional.to_tensor as F_t

Note: Replace 3.x with your specific Python version.
________________________________________
Issue 3: ParserError: while parsing a block mapping
![image](https://github.com/user-attachments/assets/74ccebbb-dfd0-42b2-9d53-0a7bb878e8c3)

Cause: The evaluation configuration file is missing paths to the model and checkpoint files. 
 
Solution:
1.	Download the pretrained model:
bash
CopyEdit
git lfs install
git clone https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Pretrained
2.	Update the video_llama_eval_withaudio.yaml file with the correct paths:
yaml
CopyEdit
llama_model: ./Video-LLaMA-2-7B-Pretrained/llama-2-7b-chat-hf
imagebind_checkpoint: ./Video-LLaMA-2-7B-Pretrained/
checkpoint_path1: /Video-LLaMA-2-7B-Pretrained/VL_LLaMA_2_7B_Pretrained.pth
checkpoint_path2: /Video-LLaMA-2-7B-Pretrained/AL_LLaMA_2_7B_Pretrained.pth
________________________________________
Issue 4: RuntimeError: Input type (float) and bias type (c10::Half) should be the same
![image](https://github.com/user-attachments/assets/de50becd-e1d3-4d5b-b253-1325d5e4aa92)

Cause: The model is attempting to use mixed precision (FP16), but the input is in FP32. 
 
Solution:
1.	Update the video_llama.yaml configuration file:
yaml
vit_precision: fp32

3.	Modify the video_llama.py script:
-> Line 54:
vit_precision = "fp32"

Remove/comment the lines 137–139 and 144 to disable loading of mixed precision models.
-> Line 549:
vit_precision = cfg.get("vit_precision", "fp32")
________________________________________
📄 Final Configuration Example
After applying all the fixes, your video_llama_eval_withaudio.yaml should resemble:

llama_model: ./Video-LLaMA-2-7B-Pretrained/llama-2-7b-chat-hf

imagebind_checkpoint: ./Video-LLaMA-2-7B-Pretrained/

checkpoint_path1: /Video-LLaMA-2-7B-Pretrained/VL_LLaMA_2_7B_Pretrained.pth

checkpoint_path2: /Video-LLaMA-2-7B-Pretrained/AL_LLaMA_2_7B_Pretrained.pth

vit_precision: fp32
________________________________________
________________________________________
👩🏻‍💻 Running the model on XPU:

To load the model on xpu run the below:
After loading the model to the CPU, we need to map the weights to the XPU using the following command:
model.to(“xpu”)
Attaching the ipynb notebook file for the reference. 
https://github.com/jaideepsai-narayan/video-analytics/blob/main/running_on_xpu.ipynb
________________________________________

🔗 Additional Resources

•	[Video-LLaMA GitHub Repository](https://github.com/DAMO-NLP-SG/Video-LLaMA)

•	[Video-LLaMA Paper: An Instruction-tuned Audio-Visual Language Model for Video UnderstandingarXiv](https://arxiv.org/abs/2306.02858)

•	[Video-LLaMA 2 Paper: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMsarXiv](https://arxiv.org/abs/2406.07476)
