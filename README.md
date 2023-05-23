# PROJET_ANDROIDE

## Installation
- Cloner le projet
- Créer un nouvel environnement et activer environnement:
```sh
conda create --name your_env python
conda activate your_env
```
- Clonner le repo google/brax en local puis lancer la commande:
```sh
cd brax
pip install --upgrade pip
pip install -e .
```
- Installation de jax pour gpu:
```
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
- Installation de diversity-algorithms:
```sh
cd diversity_algorithms_dev-master
pip install -e . 
```

## Utilisation
Pour lancer le code il faut se mettre dans le dossier diversity_algorithms_dev-master/diversity_algorithm puis lancer la commande :
```sh
python3 brax_novelty . py -e env
# ou
python3 brax_qd . py -e env
```

Les environnement disponible acutellement sont: ant-uni et ant-omni

Les paramètre réglabes sont :

    - p : la taille de la population
    - L : le nombre de step par evaluation
    - g : le nombre de génération d’évaluation
    - e : le nom de l’environnement
    - l : coefficient de descendant à générer par rapport à la population (NS)

Les notebook nov_test et qd_test sont également disponible pour manipuler les algorithmes.