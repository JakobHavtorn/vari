
echo "sudo rm -r ../vari-run-mnist2"
sudo rm -r ../vari-run-mnist2
echo "cp -r ../vari ../vari-run-mnist2"
cp -r ../vari ../vari-run-mnist2

cd ..
ls -lh
cd vari-run-mnist2
pwd

env CUDA_VISIBLE_DEVICES='4' python experiments/main_mnist.py \
--name "OOD MNISTBinarized dynamic VAE z=[64, 32] h=[[512, 512], [256, 256]] FN=0.2 IW=1 WU=0 BS=256 E=2000" with \
'n_epochs=2000' \
'batch_size=256' \
'dataset_name=MNISTBinarized' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'free_nats=0.2' \
'model_kwargs.z_dim=[64, 32]' \
'model_kwargs.h_dim=[[512, 512], [256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='4' python experiments/main_mnist.py \
--name "OOD MNISTBinarized dynamic VAE z=[64] h=[[512, 512, 256, 256]] FN=0.2 IW=1 WU=0 BS=256 E=2000" with \
'n_epochs=2000' \
'batch_size=256' \
'dataset_name=MNISTBinarized' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'free_nats=0.2' \
'model_kwargs.z_dim=[64]' \
'model_kwargs.h_dim=[[512, 512, 256, 256]]' \
# --unobserved \


env CUDA_VISIBLE_DEVICES='4' python experiments/main_mnist.py \
--name "OOD MNISTBinarized dynamic VAE z=[2] h=[[512, 512, 256, 256]] FN=0.2 IW=1 WU=0 BS=256 E=2000" with \
'n_epochs=2000' \
'batch_size=256' \
'dataset_name=MNISTBinarized' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'free_nats=0.2' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[512, 512, 256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='4' python experiments/main_mnist.py \
--name "OOD MNISTBinarized dynamic VAE z=[2, 2] h=[[512, 512], [256, 256]] FN=0.2 IW=1 WU=0 BS=256 E=2000" with \
'n_epochs=2000' \
'batch_size=256' \
'dataset_name=MNISTBinarized' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'free_nats=0.2' \
'model_kwargs.z_dim=[2, 2]' \
'model_kwargs.h_dim=[[512, 512], [256, 256]]' \
# --unobserved \




env CUDA_VISIBLE_DEVICES='4' python experiments/main_mnist.py \
--name "OOD FashionMNISTBinarized dynamic VAE z=[64] h=[[512, 512, 256, 256]] IW=1 WU=0 BS=256 E=2000" with \
'n_epochs=2000' \
'batch_size=256' \
'dataset_name=FashionMNISTBinarized' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'model_kwargs.z_dim=[64]' \
'model_kwargs.h_dim=[[512, 512, 256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='4' python experiments/main_mnist.py \
--name "OOD FashionMNISTBinarized dynamic VAE z=[64, 32] h=[[512, 512], [256, 256]] IW=1 WU=0 BS=256 E=2000" with \
'n_epochs=2000' \
'batch_size=256' \
'dataset_name=FashionMNISTBinarized' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'model_kwargs.z_dim=[64, 32]' \
'model_kwargs.h_dim=[[512, 512], [256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='4' python experiments/main_mnist.py \
--name "OOD FashionMNISTBinarized dynamic VAE z=[2] h=[[512, 512, 256, 256]] IW=1 WU=0 BS=256 E=2000" with \
'n_epochs=2000' \
'batch_size=256' \
'dataset_name=FashionMNISTBinarized' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'model_kwargs.z_dim=[2]' \
'model_kwargs.h_dim=[[512, 512, 256, 256]]' \
# --unobserved \

env CUDA_VISIBLE_DEVICES='4' python experiments/main_mnist.py \
--name "OOD FashionMNISTBinarized dynamic VAE z=[2, 2] h=[[512, 512], [256, 256]] IW=1 WU=0 BS=256 E=2000" with \
'n_epochs=2000' \
'batch_size=256' \
'dataset_name=FashionMNISTBinarized' \
'dataset_kwargs.preprocess=dynamic' \
'importance_samples=1' \
'model_kwargs.z_dim=[2, 2]' \
'model_kwargs.h_dim=[[512, 512], [256, 256]]' \
# --unobserved \