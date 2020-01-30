
echo "sudo rm -r ../vari-run-synthetic2"
sudo rm -r ../vari-run-synthetic2
echo "sudo cp -r ../vari ../vari-run-synthetic2"
sudo cp -r ../vari ../vari-run-synthetic2

cd ../vari-run-synthetic2
pwd



env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons σ=0.01 VAE FixedVariance=0.01 z=[2] h=[[64, 64, 32, 32]] IW=1 WU=0 BS=10000" with \
'n_epochs=10000' \
'batch_size=10000' \
'dataset_name=Moons' \
'dataset_kwargs.noise=0.01' \
'importance_samples=1' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[64, 64, 32, 32]]' \
'model_kwargs.decoder_distribution=["GaussianFixedVarianceLayer"]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons σ=0.01 VAE FixedVariance=0.01 z=[2, 2] h=[[64, 64], [32, 32]] IW=1 WU=0 BS=10000" with \
'n_epochs=10000' \
'batch_size=10000' \
'dataset_name=Moons' \
'dataset_kwargs.noise=0.01' \
'importance_samples=1' \
'model_kwargs.z_dim=[2, 2]' \
'model_kwargs.h_dim=[[64, 64], [32, 32]]' \
'model_kwargs.decoder_distribution=["GaussianLayer", "GaussianFixedVarianceLayer"]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons σ=0.01 VAE FixedVariance=0.01 z=[2] h=[[64, 64, 32, 32]] IW=1 WU=0 BS=256" with \
'n_epochs=10000' \
'batch_size=256' \
'dataset_name=Moons' \
'dataset_kwargs.noise=0.01' \
'importance_samples=1' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[64, 64, 32, 32]]' \
'model_kwargs.decoder_distribution=["GaussianFixedVarianceLayer"]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons σ=0.01 VAE FixedVariance=0.01 z=[2, 2] h=[[64, 64], [32, 32]] IW=1 WU=0 BS=256" with \
'n_epochs=10000' \
'batch_size=256' \
'dataset_name=Moons' \
'dataset_kwargs.noise=0.01' \
'importance_samples=1' \
'model_kwargs.z_dim=[2, 2]' \
'model_kwargs.h_dim=[[64, 64], [32, 32]]' \
'model_kwargs.decoder_distribution=["GaussianLayer", "GaussianFixedVarianceLayer"]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons σ=0.01 VAE FixedVariance=0.01 z=[2] h=[[64, 64, 32, 32]] IW=1 WU=0 BS=256" with \
'n_epochs=3000' \
'batch_size=256' \
'dataset_name=Moons' \
'dataset_kwargs.noise=0.01' \
'importance_samples=1' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[64, 64, 32, 32]]' \
'model_kwargs.decoder_distribution=["GaussianFixedVarianceLayer"]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons σ=0.01 VAE FixedVariance=0.01 z=[2, 2] h=[[64, 64], [32, 32]] IW=1 WU=0 BS=256" with \
'n_epochs=3000' \
'batch_size=256' \
'dataset_name=Moons' \
'dataset_kwargs.noise=0.01' \
'importance_samples=1' \
'model_kwargs.z_dim=[2, 2]' \
'model_kwargs.h_dim=[[64, 64], [32, 32]]' \
'model_kwargs.decoder_distribution=["GaussianLayer", "GaussianFixedVarianceLayer"]' \
# --unobserved \

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
