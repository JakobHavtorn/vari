
echo "sudo rm -r ../vari-run-synthetic2"
sudo rm -r ../vari-run-synthetic2
echo "sudo cp -r ../vari ../vari-run-synthetic2"
sudo cp -r ../vari ../vari-run-synthetic2

cd ../vari-run-synthetic2
pwd

# MOONS
env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons σ=0.01 VAE z=[2] h=[[16, 16, 8, 8]] IW=100 WU=0 BS=256" with \
'n_epochs=1000' \
'dataset_name=Moons' \
'dataset_kwargs.noise=0.01' \
'importance_samples=100' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[16, 16, 8, 8]]' \
# --unobserved

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons σ=0.01 VAE z=[2] h=[[16, 16], [8, 8]] IW=1 WU=0 BS=128" with \
'n_epochs=1000' \
'batch_size=128' \
'dataset_name=Moons' \
'dataset_kwargs.noise=0.01' \
'importance_samples=1' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[16, 16], [8, 8]]' \
# --unobserved

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons σ=0.02 VAE z=[2] h=[[64, 64, 32, 32]] IW=10 WU=0 BS=256" with \
'n_epochs=1000' \
'dataset_name=Moons' \
'dataset_kwargs.noise=0.02' \
'importance_samples=10' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[64, 64, 32, 32]]' \
# --unobserved

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons σ=0.05 VAE z=[2] h=[[128, 128, 64, 64]] IW=100 WU=0 BS=256" with \
'n_epochs=1000' \
'dataset_name=Moons' \
'dataset_kwargs.noise=0.05' \
'importance_samples=100' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[128, 128, 64, 64]]' \
--unobserved

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons σ=0.00 VAE z=[2] h=[[16, 8]] IW=100 WU=0 BS=256" with \
'n_epochs=1000' \
'dataset_name=Moons' \
'dataset_kwargs.noise=0.00' \
'importance_samples=100' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[16, 8]]' \
#--unobserved


env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons σ=0.00 VAE z=[2] h=[[32, 16]] IW=100 WU=0 BS=256" with \
'n_epochs=1000' \
'dataset_name=Moons' \
'dataset_kwargs.noise=0.00' \
'importance_samples=100' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[32, 16]]' \
#--unobserved

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons σ=0.01 VAE z=[2, 2] h=[[64, 64], [32, 32]] IW=100 WU=0 BS=256" with \
'n_epochs=1000' \
'dataset_name=Moons' \
'dataset_kwargs.noise=0.01' \
'importance_samples=100' \
'model_kwargs.z_dim=[2, 2]' \
'model_kwargs.h_dim=[[64, 64], [32, 32]]' \
# --unobserved


# # SPIRALS
# env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
# --name "OOD Spirals VAE z=[2] h=[[64, 64, 32, 32, 16, 16]] IW=10 WU=0 BS=256" with \
# 'n_epochs=1000' \
# 'dataset_name=Spirals' \
# 'importance_samples=100' \
# 'model_kwargs.z_dim=[2]' \
# 'model_kwargs.h_dim=[[64, 64, 32, 32, 16, 16]]' \
# # --unobserved

# env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
# --name "OOD Spirals VAE z=[2, 2, 2] h=[[64, 64], [32, 32], [16, 16]] IW=10 WU=0 BS=256" with \
# 'n_epochs=1000' \
# 'dataset_name=Spirals' \
# 'importance_samples=100' \
# 'model_kwargs.z_dim=[2, 2, 2]' \
# 'model_kwargs.h_dim=[[64, 64], [32, 32], [16, 16]]' \
# # --unobserved

# env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
# --name "OOD Spirals VAE z=[8, 4, 2] h=[[64, 64], [32, 32], [16, 16]] IW=10 WU=0 BS=256" with \
# 'n_epochs=1000' \
# 'dataset_name=Spirals' \
# 'importance_samples=100' \
# 'model_kwargs.z_dim=[8, 4, 2]' \
# 'model_kwargs.h_dim=[[64, 64], [32, 32], [16, 16]]' \
# # --unobserved

# env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
# --name "OOD Spirals VAE z=[2] h=[[64, 64, 32, 32, 16, 16, 8, 8]] IW=10 WU=0 BS=256" with \
# 'n_epochs=1000' \
# 'dataset_name=Spirals' \
# 'importance_samples=100' \
# 'model_kwargs.z_dim=[2]' \
# 'model_kwargs.h_dim=[[64, 64, 32, 32, 16, 16, 8, 8]]' \
# # --unobserved

# env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
# --name "OOD Spirals VAE z=[2, 2, 2, 2] h=[[64, 64], [32, 32], [16, 16], [8, 8]] IW=10 WU=0 BS=256" with \
# 'n_epochs=1000' \
# 'dataset_name=Spirals' \
# 'importance_samples=100' \
# 'model_kwargs.z_dim=[4, 2]' \
# 'model_kwargs.h_dim=[[64, 64], [32, 32], [16, 16], [8, 8]]' \
# # --unobserved

# env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
# --name "OOD Spirals VAE z=[16, 8, 4, 2] h=[[64, 64], [32, 32], [16, 16], [8, 8]] IW=10 WU=0 BS=256" with \
# 'n_epochs=1000' \
# 'dataset_name=Spirals' \
# 'importance_samples=100' \
# 'model_kwargs.z_dim=[16, 8, 4, 2]' \
# 'model_kwargs.h_dim=[[64, 64], [32, 32], [16, 16], [8, 8]]' \
# # --unobserved
