# Experiment log

## Extra layers on Hierarchical VAE for Spirals

- 3 Layers has classes are not completely separated in z3
- 4 layers has trouble activating z3 and z4.





## 24/01 Moons/Spirals

Training on Moons and Spirals without noise:

1-layer VAE z=[2] h=[64, 64, 32, 32]
2-layer VAE z=[2, 2], h=[[64, 64], [32, 32]]

- Latent pockets form in 1-layer VAE latent space
- Latent pockets are removed in z2 for 2-layer VAE which has ability to learn covariance
- Generating data from the the 2-layer model gives nicer samples with almost no wrong samples compared to 1-layer VAE
- Out-of-distribution detection is very convincing. The ELBO outside of training data is large negative numbers.
    - This almost too god - we cannot show that the 1-layer model fails at OOD detection (it actually succeeds)
    - Therefore we want to add noise to the data - in this way, we can also move the test set closer to or futher from
      the training set (before this was basically a discontinuous jump)

Training with noise = 0.05 on Moons and Spirals
- Latent pockets do not form in the 1-layer model anymore!
    - Is the noise making the task harder?
        - Adding noise to the data has made training more stable.
        - 
- ELBO and KL plots show more separation for models with additional latent layers but the difference is not remarkable.
    - This may require some tuning to be nice-looking.



noise 
- 0.05
    - [64, 64, 32, 32] --> No latent pockets
- 0.01
    - [16, 16]
    - [32, 32]
    - [64, 64]
    - [64, 64, 32, 32]



## 24/01 Moons/Spirals noise
Single layer VAE z=[2] trained on Moons and Spirals with and without noise (noise=0, noise=0,05)
6606 Spirals σ=0.05
6605 Moons σ=0.05
6602 Spirals FixedVariance
6601 Moons FixedVariance

Modellen med fixed variance/standard deviation på 0.01 træner meget bedre på Moons og Spirals end std=1. Det er også værd at bemærke at de “divergens spring” som modellen med den lærte varians havde er forsvundet med fixed varians.ELBO bliver heller ikke helt så god som før.
Med fixed varians ser jeg dog stadig at ELBO og likelihood falder og bliver <-100 for ét importance sample. Når jeg log-sum-exp’er over 100 importance samples er de begge gode.
--> Så det tyder på at variansen før blev lært til at være meget, meget lille. Det ville kunne give ustabilitet når den jævnligt når ekstremt små værdier og det giver sjældne rekonstruktioner med meget lav likelihood (<-1000).
Grunden til at variansen bliver lært så lille må være at der ikke ér nogen varians i datasættet (hvilket giver mening). Derfor prøver jeg lige at tilføje noget støj til de syntetiske datasæt og træne med lært varians igen. Jeg regner med at “divergens springene” forsvinder, men jeg ved ikke rigtig om jeg forventer at de lave likelihoods bliver løst.


## 21/01 Moons/Spirals

| ID   | Model | Model                          | Data    | KL      | IW   | ELBO (IW=1000)      |
| ---- | ----- | ------------------------------ | ------- | ------- | ---- | ------------------- |
| 6511 | VAE 2 | [64, 64, **2**, 32, 32, **2**] | Moons   | Sampled | 10   | 2.36, (7.22, 5.52)  |
| 6509 | VAE 1 | [64, 64, 32, 32, **2**]        | Moons   | Sampled | 10   | 1.95, (6.71, 4.44)  |
| 6513 | VAE 2 | [64, 64, **2**, 32, 32, **2**] | Spirals | Sampled | 10   | 1.41, (7.46, 45.69) |
| 6512 | VAE 1 | [64, 64, 32, 32, **2**]        | Spirals | Sampled | 10   | 1.02, (6.87, 5.41)  |

## 21/01 MNIST Sampled VS Analytical KL

VAE with 64 latent units.

The [512, 512] architecture matches the LadderVAE structure of building models and can be compared to ELBO=-85.10 from that paper.

ELBO's are evaluated with the same number of importance samples as during training.

| ID   | Model | Model                          | KL       | IW   | ELBO (IW=1000)          |
| ---- | ----- | ------------------------------ | -------- | ---- | ----------------------- |
| 6498 | VAE   | [512, 512, 256, 256]           | Sampled  | 10   | -82.34, (-60.81, 22.19) |
| 6503 | VAE   | [512, 512]                     | Sampled  | 10   | -84.98, (-60.15, 25.75) |
|      |       |                                |          |      |                         |
| 6501 | VAE   | [512, 512]                     | Sampled  | 1    | -88.19, (-62.14, 26.05) |
| 6531 | VAE   | [512, 512]                     | Analytic | 1    | -87.95, (-61.86, 26.09) |
|      |       |                                |          |      |                         |
| 6495 | VAE   | MNISTBinarized (deterministic) | Analytic | 10   | -54.23, (-27.60, 26.63) |
|      |       | Below have broken KL           |          |      |                         |
| 6491 | VAE   | [512, 512, 256, 256]           | Analytic | 10   | -80.62 (-61.52, 19.11)  |
| 6505 | VAE   | [512, 512]                     | Analytic | 10   | -82.85, (-60.61, 31.96) |


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