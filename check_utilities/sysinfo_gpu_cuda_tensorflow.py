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
# cd с:\my_campy\smart_city\check_utilities
# cd d:\my_campy\smart_city\check_utilities
#~~~~~~~~~~~~~~~~~~~~~~~~
# python sysinfo_gpu_cuda_tensorflow.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import tensorflow as tf
import torch

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('~'*70)
print('[INFO] GPU-CUDA-CUNN')
print(f'[INFO]     tf.config.list_physical_devices: {tf.config.list_physical_devices()}')
gpuNum = tf.config.list_physical_devices('GPU')
print(f'[INFO]     Num GPUs Available: {len(gpuNum)}')
if tf.config.list_physical_devices('GPU'):
  print('[INFO]     TensorFlow: CUDA available')
else:
  print('[INFO]     TensorFlow: CUDA is not available')
#~Задаем устройство для выполнения вычислений
device_name = tf.test.gpu_device_name()
print(f'[INFO]     Используем устройство: "{device_name}"')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('~'*70)
#~Здесь ваши вычисления
print(f'[INFO] TensorFlow(tf): {tf.__version__}')
print(f'[INFO] TensorFlow(tf)-gpu: {tf.version.VERSION}')

# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('~'*70)
print(f'[INFO] torch.version: `{torch.version}`')
print(f'[INFO] torch.cuda.is_available: `{torch.cuda.is_available()}`')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('='*70)
print('[INFO] -> program completed!')