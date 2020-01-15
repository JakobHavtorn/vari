
echo "sudo rm -r ../vari-run-synthetic"
sudo rm -r ../vari-run-synthetic
echo "sudo cp -r ../vari ../vari-run-synthetic"
sudo cp -r ../vari ../vari-run-synthetic

cd ../vari-run-synthetic
pwd

# env CUDA_VISIBLE_DEVICES='' python experiments/main_synthetic2d.py --name "OOD Moons VAE 2x[64, 64] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'warmup_epochs=0' 'learning_rate=3e-4' 'importance_samples=10' 'batch_size=256' 'dataset_name=Moons' 'vae_type=HierarchicalVariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='' python experiments/main_synthetic2d.py --name "OOD Moons VAE 2x[64, 64] IW=10 WU=50 BS=256" with 'n_epochs=1000' 'warmup_epochs=50' 'learning_rate=3e-4' 'importance_samples=10' 'batch_size=256' 'dataset_name=Moons' 'vae_type=HierarchicalVariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='' python experiments/main_synthetic2d.py --name "OOD Spirals VAE 2x[64, 64] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'warmup_epochs=0' 'learning_rate=3e-4' 'importance_samples=10' 'batch_size=256' 'dataset_name=Spirals' 'vae_type=HierarchicalVariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='' python experiments/main_synthetic2d.py --name "OOD Spirals VAE 2x[64, 64] IW=10 WU=50 BS=256" with 'n_epochs=1000' 'warmup_epochs=50' 'learning_rate=3e-4' 'importance_samples=10' 'batch_size=256' 'dataset_name=Spirals' 'vae_type=HierarchicalVariationalAutoencoder'

# env CUDA_VISIBLE_DEVICES='' python experiments/main_synthetic2d.py --name "OOD Moons VAE 1x[64, 64] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'warmup_epochs=0' 'learning_rate=3e-4' 'importance_samples=10' 'batch_size=256' 'dataset_name=Moons' 'vae_type=VariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='' python experiments/main_synthetic2d.py --name "OOD Moons VAE 1x[64, 64] IW=10 WU=50 BS=256" with 'n_epochs=1000' 'warmup_epochs=50' 'learning_rate=3e-4' 'importance_samples=10' 'batch_size=256' 'dataset_name=Moons' 'vae_type=VariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='' python experiments/main_synthetic2d.py --name "OOD Spirals VAE 1x[64, 64] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'warmup_epochs=0' 'learning_rate=3e-4' 'importance_samples=10' 'batch_size=256' 'dataset_name=Spirals' 'vae_type=VariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='' python experiments/main_synthetic2d.py --name "OOD Spirals VAE 1x[64, 64] IW=10 WU=50 BS=256" with 'n_epochs=1000' 'warmup_epochs=50' 'learning_rate=3e-4' 'importance_samples=10' 'batch_size=256' 'dataset_name=Spirals' 'vae_type=VariationalAutoencoder'

env CUDA_VISIBLE_DEVICES='' python experiments/main_synthetic2d.py --name "OOD Moons AVAE 1x[64, 64] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'warmup_epochs=0' 'learning_rate=3e-4' 'importance_samples=10' 'batch_size=256' 'dataset_name=Moons' 'vae_type=AuxilliaryVariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='' python experiments/main_synthetic2d.py --name "OOD Moons AVAE 1x[64, 64] IW=10 WU=50 BS=256" with 'n_epochs=1000' 'warmup_epochs=50' 'learning_rate=3e-4' 'importance_samples=10' 'batch_size=256' 'dataset_name=Moons' 'vae_type=AuxilliaryVariationalAutoencoder'
env CUDA_VISIBLE_DEVICES='' python experiments/main_synthetic2d.py --name "OOD Spirals AVAE 1x[64, 64] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'warmup_epochs=0' 'learning_rate=3e-4' 'importance_samples=10' 'batch_size=256' 'dataset_name=Spirals' 'vae_type=AuxilliaryVariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='' python experiments/main_synthetic2d.py --name "OOD Spirals AVAE 1x[64, 64] IW=10 WU=50 BS=256" with 'n_epochs=1000' 'warmup_epochs=50' 'learning_rate=3e-4' 'importance_samples=10' 'batch_size=256' 'dataset_name=Spirals' 'vae_type=AuxilliaryVariationalAutoencoder'
