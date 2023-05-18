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

## Test
Pour lancer le code il faut se mettre dans le dossier diversity_algorithms_dev-master/diversity_algorithms/experiments puis lancer la commande:

```sh
python3 brax_novelty.py -e ant 
# ou 
python3 brax_qd.py -e ant
```

Pour reproduire l'erreur il faut décommenter les lignes de code de mise à jour des statistiques dans les fichiers situé dans le dossier diversity_algorithms_dev-master/diversity_algorithms/algorithms:
 - novelty_search.py: lignes 192-195 et lignes 313-317
 - quality_diversity.py: lignes 380-383 et lignes 495-498
 
 
Consigne: 
 
QDAX
NS -> couverture de B
QD -> couverture et QDSCORE et indiv fitmax en fonction nombre evaluation
taille de pop différente
taille du réseau de neuronnes
cpu vs gpu
fixe budget eval différente
se comparé avec QDAX discussion
se comparé avec diversity algo

nslc pour archive non structuré, récupère les point situé sur du front de pareto sur la pop est les ajoute
recherche sur internet pour comparer avec cpu
prendre un ratio pour ns et le justifier
noter tout les point qui pose problème, essayer de trouver ce qui pose problème et donner des pistes d'amélioration
citer les source dans les entete de fonction, regarder les licences
22 mai 13H