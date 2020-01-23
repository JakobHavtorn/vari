
echo "sudo rm -r ../vari-run-synthetic"
sudo rm -r ../vari-run-synthetic
echo "sudo cp -r ../vari ../vari-run-synthetic"
sudo cp -r ../vari ../vari-run-synthetic

cd ../vari-run-synthetic
pwd

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons VAE z=[2, 2, 2] IW=10 WU=0 BS=256" with \
'n_epochs=1000' \
'dataset_name=Moons' \
'importance_samples=100' \
'model_kwargs.z_dim=[2, 2, 2]' \
'model_kwargs.h_dim=[[64, 64], [32, 32], [16, 16]]' \
'model_kwargs.encoder_distribution=["GaussianLayer", "GaussianLayer", "GaussianLayer"]' \
'model_kwargs.decoder_distribution=["GaussianLayer", "GaussianLayer", "GaussianLayer"]' \
# --unobserved \
&& \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons VAE z=[2, 2] IW=10 WU=0 BS=256" with \
'n_epochs=1000' \
'dataset_name=Moons' \
'importance_samples=100' \
'model_kwargs.z_dim=[2, 2]' \
'model_kwargs.h_dim=[[64, 64], [32, 32]]' \
'model_kwargs.encoder_distribution=["GaussianLayer", "GaussianLayer"]' \
'model_kwargs.decoder_distribution=["GaussianLayer", "GaussianLayer"]' \
# --unobserved \
&& \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Moons VAE z=[2] IW=10 WU=0 BS=256" with \
'n_epochs=1000' \
'dataset_name=Moons' \
'importance_samples=100' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[64, 64]]' \
'model_kwargs.encoder_distribution=["GaussianLayer"]' \
'model_kwargs.decoder_distribution=["GaussianLayer"]' \
# --unobserved \
&& \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Spirals VAE z=[8, 4, 2] IW=10 WU=0 BS=256" with \
'n_epochs=1000' \
'dataset_name=Spirals' \
'importance_samples=100' \
'model_kwargs.z_dim=[8, 4, 2]' \
'model_kwargs.h_dim=[[64, 64], [32, 32], [16, 16]]' \
'model_kwargs.encoder_distribution=["GaussianLayer", "GaussianLayer", "GaussianLayer"]' \
'model_kwargs.decoder_distribution=["GaussianLayer", "GaussianLayer", "GaussianLayer"]' \
# --unobserved \
&& \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Spirals VAE z=[4, 2] IW=10 WU=0 BS=256" with \
'n_epochs=1000' \
'dataset_name=Spirals' \
'importance_samples=100' \
'model_kwargs.z_dim=[4, 2]' \
'model_kwargs.h_dim=[[64, 64], [32, 32]]' \
'model_kwargs.encoder_distribution=["GaussianLayer", "GaussianLayer"]' \
'model_kwargs.decoder_distribution=["GaussianLayer", "GaussianLayer"]' \
# --unobserved \
&& \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Spirals VAE z=[2, 2, 2] IW=10 WU=0 BS=256" with \
'n_epochs=1000' \
'dataset_name=Spirals' \
'importance_samples=100' \
'model_kwargs.z_dim=[2, 2, 2]' \
'model_kwargs.h_dim=[[64, 64], [32, 32], [16, 16]]' \
'model_kwargs.encoder_distribution=["GaussianLayer", "GaussianLayer", "GaussianLayer"]' \
'model_kwargs.decoder_distribution=["GaussianLayer", "GaussianLayer", "GaussianLayer"]' \
# --unobserved \
&& \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Spirals VAE z=[2, 2] IW=10 WU=0 BS=256" with \
'n_epochs=1000' \
'dataset_name=Spirals' \
'importance_samples=100' \
'model_kwargs.z_dim=[2, 2]' \
'model_kwargs.h_dim=[[64, 64], [32, 32]]' \
'model_kwargs.encoder_distribution=["GaussianLayer", "GaussianLayer"]' \
'model_kwargs.decoder_distribution=["GaussianLayer", "GaussianLayer"]' \
# --unobserved \
&& \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_synthetic2d.py \
--name "OOD Spirals VAE z=[2] IW=10 WU=0 BS=256" with \
'n_epochs=1000' \
'dataset_name=Spirals' \
'importance_samples=100' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[64, 64]]' \
'model_kwargs.encoder_distribution=["GaussianLayer"]' \
'model_kwargs.decoder_distribution=["GaussianLayer"]' \
# --unobserved \
# && \
