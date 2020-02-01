
echo "sudo rm -r ../vari-run-mnist2"
sudo rm -r ../vari-run-mnist2
echo "cp -r ../vari ../vari-run-mnist2"
cp -r ../vari ../vari-run-mnist2

cd ..
ls -lh
cd vari-run-mnist2
pwd


env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py \
--name "OOD MNISTBinarized deterministic VAE z=[2] h=[[512, 512, 256, 256]] IW=1 WU=0 BS=256" with \
'n_epochs=1000' \
'batch_size=256' \
'dataset_name=MNISTBinarized' \
'dataset_kwargs.preprocess=deterministic' \
'importance_samples=1' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[512, 512, 256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py \
--name "OOD MNISTBinarized deterministic VAE z=[5, 2] h=[[512, 512], [256, 256]] IW=1 WU=0 BS=256" with \
'n_epochs=1000' \
'batch_size=256' \
'dataset_name=MNISTBinarized' \
'dataset_kwargs.preprocess=deterministic' \
'importance_samples=1' \
'model_kwargs.z_dim=[5, 2]' \
'model_kwargs.h_dim=[[512, 512], [256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py \
--name "OOD FashionMNISTBinarized deterministic VAE z=[2] h=[[512, 512, 256, 256]] IW=1 WU=0 BS=256" with \
'n_epochs=1000' \
'batch_size=256' \
'dataset_name=FashionMNISTBinarized' \
'dataset_kwargs.preprocess=deterministic' \
'importance_samples=1' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[512, 512, 256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py \
--name "OOD FashionMNISTBinarized deterministic VAE z=[5, 2] h=[[512, 512], [256, 256]] IW=1 WU=0 BS=256" with \
'n_epochs=1000' \
'batch_size=256' \
'dataset_name=FashionMNISTBinarized' \
'dataset_kwargs.preprocess=deterministic' \
'importance_samples=1' \
'model_kwargs.z_dim=[5, 2]' \
'model_kwargs.h_dim=[[512, 512], [256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py \
--name "OOD MNISTBinarized deterministic VAE z=[2] h=[[512, 512, 256, 256]] IW=10 WU=0 BS=256" with \
'n_epochs=1000' \
'batch_size=256' \
'dataset_name=MNISTBinarized' \
'dataset_kwargs.preprocess=deterministic' \
'importance_samples=10' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[512, 512, 256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py \
--name "OOD MNISTBinarized deterministic VAE z=[5, 2] h=[[512, 512], [256, 256]] IW=10 WU=0 BS=256" with \
'n_epochs=1000' \
'batch_size=256' \
'dataset_name=MNISTBinarized' \
'dataset_kwargs.preprocess=deterministic' \
'importance_samples=10' \
'model_kwargs.z_dim=[5, 2]' \
'model_kwargs.h_dim=[[512, 512], [256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py \
--name "OOD FashionMNISTBinarized deterministic VAE z=[2] h=[[512, 512, 256, 256]] IW=10 WU=0 BS=256" with \
'n_epochs=1000' \
'batch_size=256' \
'dataset_name=FashionMNISTBinarized' \
'dataset_kwargs.preprocess=deterministic' \
'importance_samples=10' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[512, 512, 256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='3' python experiments/main_mnist.py \
--name "OOD FashionMNISTBinarized deterministic VAE z=[5, 2] h=[[512, 512], [256, 256]] IW=10 WU=0 BS=256" with \
'n_epochs=1000' \
'batch_size=256' \
'dataset_name=FashionMNISTBinarized' \
'dataset_kwargs.preprocess=deterministic' \
'importance_samples=10' \
'model_kwargs.z_dim=[5, 2]' \
'model_kwargs.h_dim=[[512, 512], [256, 256]]' \
# --unobserved \
