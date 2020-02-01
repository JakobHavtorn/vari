
echo "sudo rm -r ../vari-run-synthetic2"
sudo rm -r ../vari-run-synthetic2
echo "sudo cp -r ../vari ../vari-run-synthetic2"
sudo cp -r ../vari ../vari-run-synthetic2

cd ../vari-run-synthetic2
pwd



env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons σ=0.01 VAE ELU z=[2] h=[[64, 64, 32, 32]] IW=1 WU=0 BS=256" with \
'n_epochs=1000' \
'batch_size=256' \
'dataset_name=Moons' \
'dataset_kwargs.noise=0.01' \
'importance_samples=1' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[64, 64, 32, 32]]' \
'model_kwargs.decoder_distribution=["GaussianLayer"]' \
# --unobserved \


env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons σ=0.01 VAE ELU z=[2] h=[[64, 64], [32, 32]] IW=1 WU=0 BS=256" with \
'n_epochs=1000' \
'batch_size=256' \
'dataset_name=Moons' \
'dataset_kwargs.noise=0.01' \
'importance_samples=1' \
'model_kwargs.z_dim=[2, 2]' \
'model_kwargs.h_dim=[[64, 64], [32, 32]]' \
'model_kwargs.decoder_distribution=["GaussianLayer", "GaussianLayer"]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons σ=0.00 VAE ELU z=[2] h=[[64, 64, 32, 32]] IW=1 WU=0 BS=256" with \
'n_epochs=1000' \
'batch_size=256' \
'dataset_name=Moons' \
'dataset_kwargs.noise=0.00' \
'importance_samples=1' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[64, 64, 32, 32]]' \
'model_kwargs.decoder_distribution=["GaussianLayer"]' \
# --unobserved \


env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons σ=0.00 VAE ELU z=[2] h=[[64, 64], [32, 32]] IW=1 WU=0 BS=256" with \
'n_epochs=1000' \
'batch_size=256' \
'dataset_name=Moons' \
'dataset_kwargs.noise=0.00' \
'importance_samples=1' \
'model_kwargs.z_dim=[2, 2]' \
'model_kwargs.h_dim=[[64, 64], [32, 32]]' \
'model_kwargs.decoder_distribution=["GaussianLayer", "GaussianLayer"]' \
# --unobserved \



# SPIRALS
env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Spirals σ=0.01 VAE ELU z=[2] h=[[64, 64, 32, 32]] IW=1 WU=0 BS=256" with \
'n_epochs=1000' \
'batch_size=256' \
'dataset_name=Spirals' \
'dataset_kwargs.noise=0.01' \
'importance_samples=1' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[64, 64, 32, 32]]' \
'model_kwargs.decoder_distribution=["GaussianLayer"]' \
# --unobserved \


env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Spirals σ=0.01 VAE ELU z=[2] h=[[64, 64], [32, 32]] IW=1 WU=0 BS=256" with \
'n_epochs=1000' \
'batch_size=256' \
'dataset_name=Spirals' \
'dataset_kwargs.noise=0.01' \
'importance_samples=1' \
'model_kwargs.z_dim=[2, 2]' \
'model_kwargs.h_dim=[[64, 64], [32, 32]]' \
'model_kwargs.decoder_distribution=["GaussianLayer", "GaussianLayer"]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Spirals σ=0.00 VAE ELU z=[2] h=[[64, 64, 32, 32]] IW=1 WU=0 BS=256" with \
'n_epochs=1000' \
'batch_size=256' \
'dataset_name=Spirals' \
'dataset_kwargs.noise=0.00' \
'importance_samples=1' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[64, 64, 32, 32]]' \
'model_kwargs.decoder_distribution=["GaussianLayer"]' \
# --unobserved \


env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Spirals σ=0.00 VAE ELU z=[2] h=[[64, 64], [32, 32]] IW=1 WU=0 BS=256" with \
'n_epochs=1000' \
'batch_size=256' \
'dataset_name=Spirals' \
'dataset_kwargs.noise=0.00' \
'importance_samples=1' \
'model_kwargs.z_dim=[2, 2]' \
'model_kwargs.h_dim=[[64, 64], [32, 32]]' \
'model_kwargs.decoder_distribution=["GaussianLayer", "GaussianLayer"]' \
# --unobserved \