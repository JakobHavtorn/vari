
echo "sudo rm -r ../vari-run-mnist2"
sudo rm -r ../vari-run-mnist2
echo "cp -r ../vari ../vari-run-mnist2"
cp -r ../vari ../vari-run-mnist2

cd ..
ls -lh
cd vari-run-mnist2
pwd



# BEST EXPERIMENTS

env CUDA_VISIBLE_DEVICES='6' python experiments/main_mnist.py \
--name "OOD MNISTContinuous dynamic VAE Beta(x) z=[64] h=[[512, 512, 256, 256]] FN=0.2 IW=1 WU=0 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=MNISTContinuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=0' \
'free_nats=0.2' \
'build_kwargs.skip_connections=False' \
'build_kwargs.z_dim=[64]' \
'build_kwargs.h_dim=[[512, 512, 256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='4' python experiments/main_mnist.py \
--name "OOD MNISTContinuous dynamic VAE Beta(x) z=[64] h=[[512, 512, 256, 256, 128, 128]] FN=0.2 IW=1 WU=0 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=MNISTContinuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=0' \
'free_nats=0.2' \
'build_kwargs.skip_connections=False' \
'build_kwargs.z_dim=[64]' \
'build_kwargs.h_dim=[[512, 512, 256, 256, 128, 128]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='6' python experiments/main_mnist.py \
--name "OOD MNISTContinuous dynamic VAE Beta(x) z=[64, 32] h=[[512, 512], [256, 256]] FN=0.2 IW=1 WU=200 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=MNISTContinuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=200' \
'free_nats=0.2' \
'build_kwargs.skip_connections=False' \
'build_kwargs.z_dim=[64, 32]' \
'build_kwargs.h_dim=[[512, 512], [256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='6' python experiments/main_mnist.py \
--name "OOD MNISTContinuous dynamic VAE Beta(x) z=[64, 32, 16] h=[[512, 512], [256, 256], [128, 128]] FN=0.2 IW=1 WU=200 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=MNISTContinuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=200' \
'free_nats=0.2' \
'build_kwargs.skip_connections=False' \
'build_kwargs.z_dim=[64, 32, 16]' \
'build_kwargs.h_dim=[[512, 512], [256, 256], [128, 128]]' \
# --unobserved \




env CUDA_VISIBLE_DEVICES='4' python experiments/main_mnist.py \
--name "OOD FashionMNISTContinuous dynamic VAE Beta(x) z=[2] h=[[512, 512, 256, 256]] FN=0.2 IW=1 WU=0 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=FashionMNISTContinuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=0' \
'free_nats=0.2' \
'build_kwargs.skip_connections=False' \
'build_kwargs.z_dim=[2]' \
'build_kwargs.h_dim=[[512, 512, 256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='4' python experiments/main_mnist.py \
--name "OOD FashionMNISTContinuous dynamic VAE Beta(x) z=[8, 2] h=[[512, 512], [256, 256]] FN=0.2 IW=1 WU=0 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=FashionMNISTContinuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=0' \
'free_nats=0.2' \
'build_kwargs.skip_connections=False' \
'build_kwargs.z_dim=[8, 2]' \
'build_kwargs.h_dim=[[512, 512], [256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='4' python experiments/main_mnist.py \
--name "OOD FashionMNISTContinuous dynamic VAE Beta(x) z=[32, 16, 2, 2] h=[[256, 256], [128, 128], [64, 64], [32, 32]] FN=0.2 IW=1 WU=200 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=FashionMNISTContinuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=200' \
'free_nats=0.2' \
'build_kwargs.skip_connections=False' \
'build_kwargs.z_dim=[32, 16, 2, 2]' \
'build_kwargs.h_dim=[[256, 256], [128, 128], [64, 64], [32, 32]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='4' python experiments/main_mnist.py \
--name "OOD FashionMNISTContinuous dynamic VAE Beta(x) z=[64] h=[[512, 512, 256, 256, 128, 128]] FN=0.2 IW=1 WU=0 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=FashionMNISTContinuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=0' \
'free_nats=0.2' \
'build_kwargs.skip_connections=False' \
'build_kwargs.z_dim=[64]' \
'build_kwargs.h_dim=[[512, 512, 256, 256, 128, 128]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='4' python experiments/main_mnist.py \
--name "OOD FashionMNISTContinuous dynamic VAE Beta(x) z=[64, 32] h=[[512, 512], [256, 256]] FN=0.2 IW=1 WU=200 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=FashionMNISTContinuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=200' \
'free_nats=0.2' \
'build_kwargs.skip_connections=False' \
'build_kwargs.z_dim=[64, 32]' \
'build_kwargs.h_dim=[[512, 512], [256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='4' python experiments/main_mnist.py \
--name "OOD FashionMNISTContinuous dynamic VAE Beta(x) z=[64, 32, 16] h=[[512, 512], [256, 256], [128, 128]] FN=0.2 IW=1 WU=200 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=FashionMNISTContinuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=200' \
'free_nats=0.2' \
'build_kwargs.skip_connections=False' \
'build_kwargs.z_dim=[64, 32, 16]' \
'build_kwargs.h_dim=[[512, 512], [256, 256], [128, 128]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='4' python experiments/main_mnist.py \
--name "OOD FashionMNISTBinarized dynamic VAE Beta(x) z=[64, 32, 16, 8] h=[[512, 512], [256, 256], [128, 128], [64, 64]] FN=0.2 IW=1 WU=200 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=FashionMNISTBinarized' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=200' \
'free_nats=0.2' \
'build_kwargs.skip_connections=True' \
'build_kwargs.z_dim=[64, 32, 16, 8]' \
'build_kwargs.h_dim=[[512, 512], [256, 256], [128, 128], [64, 64]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='6' python experiments/main_mnist.py \
--name "OOD FashionMNISTContinuous dynamic VAE BN Beta(x) z=[64, 32, 16, 8, 4] h=[[512, 512], [256, 256], [128, 128], [64, 64], [32, 32]] FN=0.2 IW=1 WU=200 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=FashionMNISTContinuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=200' \
'free_nats=0.2' \
'build_kwargs.batchnorm=True' \
'build_kwargs.skip_connections=True' \
'build_kwargs.z_dim=[64, 32, 16, 8, 4]' \
'build_kwargs.h_dim=[[512, 512], [256, 256], [128, 128], [64, 64], [32, 32]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='6' python experiments/main_mnist.py \
--name "OOD FashionMNISTContinuous dynamic VAE BN Beta(x) z=[64, 32, 16, 8, 4, 2] h=[[512, 512], [256, 256], [128, 128], [64, 64], [32, 32], [16, 16]] FN=0.2 IW=1 WU=200 BS=256 E=2500" with \
'n_epochs=2500' \
'batch_size=256' \
'dataset_name=FashionMNISTContinuous' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'warmup_epochs=200' \
'free_nats=0.2' \
'build_kwargs.batchnorm=True' \
'build_kwargs.skip_connections=True' \
'build_kwargs.z_dim=[64, 32, 16, 8, 4, 2]' \
'build_kwargs.h_dim=[[512, 512], [256, 256], [128, 128], [64, 64], [32, 32], [16, 16]]' \
# --unobserved \
