# Experiment log

## 15/01 MNIST/FashionMNIST

**Full datasets**

| ID       | Model | Train dataset |
| -------- | ----- | ------------- |
| 6243     | VAE   | MNIST         |
| 6253     | VAE   | FashionMNIST  |
| 6278     | AVAE  | MNIST         |
| 6286     | AVAE  | FashionMNIST  |
| 6263     | HVAE  | MNIST         |
| 6256     | HVAE  | FashionMNIST  |

**Excluded labels**

| ID       | Model | EXCL | Train dataset |
| -------- | ----- | ---- | ------------- |
| 6284     | AVAE  | 8    | FashionMNIST  |
| 6264     | HVAE  | 8    | FashionMNIST  |
| 6255     | VAE   | 8    | FashionMNIST  |
| 6252     | AVAE  | 8    | MNIST         |
| 6251     | HVAE  | 8    | MNIST         |
| 6250     | VAE   | 8    | MNIST         |
| 6249     | AVAE  | 0    | FashionMNIST  |
| 6248     | HVAE  | 0    | FashionMNIST  |
| 6247     | VAE   | 0    | FashionMNIST  |
| 6246     | AVAE  | 4    | MNIST         |
| 6245     | HVAE  | 4    | MNIST         |
| 6244     | VAE   | 4    | MNIST         |

## 17-01 Moons/Spirals

| ID       | Model | Train dataset |
| -------- | ----- | ------------- |
| 6372     | VAE   | Moons         |
| 6378     | HVAE  | Moons         |
| 6380     | AVAE  | Moons         |
| 6383     | VAE   | Spirals       |
| 6386     | HVAE  | Spirals       |
| 6387     | AVAE  | Spirals       |


## Moons and Spiral 

if dataset_name == 'Moons':
    run_ids = dict(
        vae=6209,
        hvae=6203,
        avae=6219,
    )
elif dataset_name == 'Spirals':
    run_ids = dict(
        vae=6211,
        avae=6220,
        hvae=6205,
    )