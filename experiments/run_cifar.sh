env CUDA_VISIBLE_DEVICES='9' python experiments/main_mnist.py \
--name "OOD SVHNContinuous dynamic VAE Beta(x) z=[64] h=[[512, 512, 256, 256]] FN=0.2 IW=1 WU=0 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=SVHNContinuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=0' \
'free_nats=0.2' \
'build_kwargs.skip_connections=False' \
'build_kwargs.z_dim=[64]' \
'build_kwargs.h_dim=[[512, 512, 256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='9' python experiments/main_mnist.py \
--name "OOD SVHNContinuous dynamic VAE Beta(x) z=[64, 32] h=[[512, 512], [256, 256]] FN=0.2 IW=1 WU=0 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=SVHNContinuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=0' \
'free_nats=0.2' \
'build_kwargs.skip_connections=False' \
'build_kwargs.z_dim=[64, 32]' \
'build_kwargs.h_dim=[[512, 512], [256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='9' python experiments/main_mnist.py \
--name "OOD SVHNContinuous dynamic VAE Beta(x) z=[64, 32, 16] h=[[512, 512], [256, 256], [128., 128]] FN=0.2 IW=1 WU=0 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=SVHNContinuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=0' \
'free_nats=0.2' \
'build_kwargs.skip_connections=False' \
'build_kwargs.z_dim=[64, 32, 16]' \
'build_kwargs.h_dim=[[512, 512], [256, 256], [128., 128]]' \
# --unobserved \



env CUDA_VISIBLE_DEVICES='9' python experiments/main_mnist.py \
--name "OOD CIFAR10Continuous dynamic VAE RFL Beta(x) z=[64, 32, 16] h=[[512, 512], [256, 256], [128, 128]] FN=0.2 IW=1 WU=200 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=CIFAR10Continuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=200' \
'free_nats=0.2' \
'build_kwargs.skip_connections=False' \
'build_kwargs.z_dim=[64, 32, 16]' \
'build_kwargs.h_dim=[[512, 512], [256, 256], [128, 128]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='9' python experiments/main_mnist.py \
--name "OOD CIFAR10Continuous dynamic VAE Skip Beta(x) z=[64, 32, 16, 8] h=[[512, 512], [256, 256], [128, 128], [64, 64]] FN=0.2 IW=1 WU=300 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=CIFAR10Continuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=300' \
'free_nats=0.2' \
'build_kwargs.skip_connections=True' \
'build_kwargs.z_dim=[64, 32, 16, 8]' \
'build_kwargs.h_dim=[[512, 512], [256, 256], [128, 128], [64, 64]]' \
# --unobserved \


env CUDA_VISIBLE_DEVICES='9' python experiments/main_mnist.py \
--name "OOD CIFAR10Continuous dynamic VAE Skip Beta(x) z=[64, 32, 16, 8, 4] h=[[512, 512], [256, 256], [128, 128], [64, 64], [32, 32]] FN=0.2 IW=1 WU=0 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=CIFAR10Continuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=0' \
'free_nats=0.2' \
'build_kwargs.skip_connections=False' \
'build_kwargs.z_dim=[64, 32, 16, 8, 4]' \
'build_kwargs.h_dim=[[512, 512], [256, 256], [128, 128], [64, 64], [32, 32]]' \
# --unobserved \



env CUDA_VISIBLE_DEVICES='9' python experiments/main_mnist.py \
--name "OOD CIFAR10Continuous dynamic VAE RFL Beta(x) z=[128] h=[[512, 512, 256, 256, 128, 128]] FN=0.2 IW=1 WU=0 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=CIFAR10Continuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=0' \
'free_nats=0.2' \
'build_kwargs.skip_connections=False' \
'build_kwargs.z_dim=[128]' \
'build_kwargs.h_dim=[[512, 512, 256, 256, 128, 128]]' \
# --unobserved 


env CUDA_VISIBLE_DEVICES='9' python experiments/main_mnist.py \
--name "OOD CIFAR10Continuous dynamic VAE RFL Beta(x) z=[128, 64, 32] h=[[512, 512], [256, 256], [128, 128]] FN=0.2 IW=1 WU=200 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=CIFAR10Continuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=200' \
'free_nats=0.2' \
'build_kwargs.skip_connections=False' \
'build_kwargs.z_dim=[128, 64, 32]' \
'build_kwargs.h_dim=[[512, 512], [256, 256], [128, 128]]' \
# --unobserved \
