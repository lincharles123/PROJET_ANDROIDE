# PROJET_ANDROIDE
Pour lancer le code il faut se mettre dans le dossier diversity_algorithms_dev-master/diversity_algorithms/experiments puis lancer la commande:

python3 brax_novelty.py -e antoupython3 brax_qd.py -e ant

Pour reproduire l'erreur il faut décommenter les lignes de code de mise à jour des statistiques dans les fichiers situé dans le dossier diversity_algorithms_dev-master/diversity_algorithms/algorithms:
 - novelty_search.py: lignes 192-195 et lignes 313-317
 - quality_diversity.py: lignes 380-383 et lignes 495-498
