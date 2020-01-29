
echo "sudo rm -r ../vari-run-mnist2"
sudo rm -r ../vari-run-mnist2
echo "cp -r ../vari ../vari-run-mnist2"
cp -r ../vari ../vari-run-mnist2

cd ..
ls -lh
cd vari-run-mnist2
pwd


# Full FashionMNIST and MNIST datasets, VAE, 2-layer VAE and AVAE models
# env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD master MNISTBinarized VAE [2] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'warmup_epochs=0' 'dataset_name=MNISTBinarized' 'vae_type=VariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD master FashionMNISTBinarized VAE [2] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'warmup_epochs=0' 'dataset_name=FashionMNISTBinarized' 'vae_type=VariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD master MNISTBinarized VAE [5, 2] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'warmup_epochs=0' 'dataset_name=MNISTBinarized' 'vae_type=HierarchicalVariationalAutoencoder'
env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD master FashionMNISTBinarized VAE [5, 2] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'warmup_epochs=0' 'dataset_name=FashionMNISTBinarized' 'vae_type=HierarchicalVariationalAutoencoder'
env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD master MNISTBinarized AVAE [2] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'warmup_epochs=0' 'dataset_name=MNISTBinarized' 'vae_type=AuxilliaryVariationalAutoencoder'
env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD master FashionMNISTBinarized AVAE [2] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'warmup_epochs=0' 'dataset_name=FashionMNISTBinarized' 'vae_type=AuxilliaryVariationalAutoencoder'


# Excluded labels FashionMNIST and MNIST datasets, VAE, 2-layer VAE and AVAE models
# env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD MNISTBinarized VAE [2] EXCL=[4] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'exclude_labels=[4]' 'warmup_epochs=0' 'dataset_name=MNISTBinarized' 'vae_type=VariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD MNISTBinarized VAE [5, 2] EXCL=[4] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'exclude_labels=[4]' 'warmup_epochs=0' 'dataset_name=MNISTBinarized' 'vae_type=HierarchicalVariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD MNISTBinarized AVAE [2] EXCL=[4] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'exclude_labels=[4]' 'warmup_epochs=0' 'dataset_name=MNISTBinarized' 'vae_type=AuxilliaryVariationalAutoencoder'

# env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD FashionMNISTBinarized VAE [2] EXCL=[0] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'exclude_labels=[0]' 'warmup_epochs=0' 'dataset_name=FashionMNISTBinarized' 'vae_type=VariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD FashionMNISTBinarized VAE [5, 2] EXCL=[0] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'exclude_labels=[0]' 'warmup_epochs=0' 'dataset_name=FashionMNISTBinarized' 'vae_type=HierarchicalVariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD FashionMNISTBinarized AVAE [2] EXCL=[0] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'exclude_labels=[0]' 'warmup_epochs=0' 'dataset_name=FashionMNISTBinarized' 'vae_type=AuxilliaryVariationalAutoencoder'

# env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD MNISTBinarized VAE [2] EXCL=[8] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'warmup_epochs=0' 'dataset_name=MNISTBinarized' 'vae_type=VariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD MNISTBinarized VAE [5, 2] EXCL=[8] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'exclude_labels=[8]' 'warmup_epochs=0' 'dataset_name=MNISTBinarized' 'vae_type=HierarchicalVariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD MNISTBinarized AVAE [2] EXCL=[8] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'exclude_labels=[8]' 'warmup_epochs=0' 'dataset_name=MNISTBinarized' 'vae_type=AuxilliaryVariationalAutoencoder'

# env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD FashionMNISTBinarized VAE [2] EXCL=[8] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'exclude_labels=[8]' 'warmup_epochs=0' 'dataset_name=FashionMNISTBinarized' 'vae_type=VariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD FashionMNISTBinarized VAE [5, 2] EXCL=[8] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'exclude_labels=[8]' 'warmup_epochs=0' 'dataset_name=FashionMNISTBinarized' 'vae_type=HierarchicalVariationalAutoencoder'
# env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py --name "OOD FashionMNISTBinarized AVAE [2] EXCL=[8] IW=10 WU=0 BS=256" with 'n_epochs=1000' 'exclude_labels=[8]' 'warmup_epochs=0' 'dataset_name=FashionMNISTBinarized' 'vae_type=AuxilliaryVariationalAutoencoder'
