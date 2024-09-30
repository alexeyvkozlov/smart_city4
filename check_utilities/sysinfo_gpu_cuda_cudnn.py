#~~~~~~~~~~~~~~~~~~~~~~~~
#~ узнать драйвер NVIDIA, в терминале:
# nvcc -V
# pip show torch
# pip show torchvision
# pip show tensorflow
#~~~~~~~~~~~~~~~~~~~~~~~~
#~ check on GPU
#~ !nvidia-smi
#~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd c:\my_campy\smart_city\check_utilities
# cd d:\my_campy\smart_city\check_utilities\
#~~~~~~~~~~~~~~~~~~~~~~~~
# python sysinfo_gpu_cuda_cudnn.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with
#https://stackoverflow.com/questions/57814535/assertionerror-torch-not-compiled-with-cuda-enabled-in-spite-upgrading-to-cud
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#https://pytorch.org/get-started/locally/
#https://pytorch.org/get-started/locally/
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#https://pytorch.org/get-started/previous-versions/
#https://pytorch.org/get-started/previous-versions/
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#https://pytorch.org/get-started/locally/
#https://pytorch.org/get-started/previous-versions/
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#https://pytorch.org/get-started/locally/
#https://pytorch.org/get-started/previous-versions/
# CUDA 11.8
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
#pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CUDA 11.1
#pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# CUDA 11.1
#pip install torch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1
#pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# CUDA 11.3
#conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import torch

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('~'*70)
x = torch.rand(5, 3)
print(x)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#>>> import torch
#>>> torch.cuda.is_available()
print('~'*70)
print(f'[INFO] torch.cuda.is_available(): `{torch.cuda.is_available()}`')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#>>> import torch
#>>> torch.zeros(1).cuda()
print('~'*70)
print(f'[INFO] torch.zeros(1).cuda(): `{torch.zeros(1).cuda()}`')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('~'*70)
print(f'[INFO] torch.version: `{torch.version}`')
print(f'[INFO] torch.cuda.is_available: `{torch.cuda.is_available()}`')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('~'*70)
#~
print('[INFO] если версия cuDNN отображается, значит, установка прошла успешно:')
print(torch.backends.cudnn.version())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('='*70)
print('[INFO] -> program completed!')