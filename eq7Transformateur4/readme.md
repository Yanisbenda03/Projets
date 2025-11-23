# Simulation d’un Marché du Cacao – Groupe Transformateur

## 1. Description du projet

Ce projet simule un marché du cacao composé de plusieurs acteurs économiques :

- Producteurs  
- Transformateurs  
- Distributeurs  

Chaque agent prend des décisions (achat, vente, production, négociation) pour maximiser ses résultats dans un marché concurrentiel.

Notre groupe jouait le rôle des **Transformateurs** : acheter du cacao brut, le transformer et le revendre aux distributeurs.  
L’objectif était d’optimiser la marge en gérant les volumes, les prix, les contrats et les aléas du marché.

## 2. Rôle des Transformateurs

Le Transformateur est l’intermédiaire entre les producteurs et les distributeurs.

Responsabilités principales :

- analyser et sélectionner les offres des producteurs,  
- acheter via différents mécanismes (bourse, contrat cadre, appel d’offres),  
- transformer le cacao brut en produits dérivés,  
- vendre la production aux distributeurs,  
- ajuster les décisions selon les stocks, la demande et l’évolution des prix.

## 3. Structure du code

# Simulation d’un Marché du Cacao – Groupe Transformateur

## 1. Description du projet

Ce projet simule un marché du cacao composé de plusieurs acteurs économiques :

- Producteurs  
- Transformateurs  
- Distributeurs  

Chaque agent prend des décisions (achat, vente, production, négociation) pour maximiser ses résultats dans un marché concurrentiel.

Notre groupe jouait le rôle des **Transformateurs** : acheter du cacao brut, le transformer et le revendre aux distributeurs.  
L’objectif était d’optimiser la marge en gérant les volumes, les prix, les contrats et les aléas du marché.

## 2. Rôle des Transformateurs

Le Transformateur est l’intermédiaire entre les producteurs et les distributeurs.

Responsabilités principales :

- analyser et sélectionner les offres des producteurs,  
- acheter via différents mécanismes (bourse, contrat cadre, appel d’offres),  
- transformer le cacao brut en produits dérivés,  
- vendre la production aux distributeurs,  
- ajuster les décisions selon les stocks, la demande et l’évolution des prix.


## 3. Structure du code

- **Transformateur4.java** : logique principale du transformateur  
- **Transformateur4Acteur.java** : comportement général de l’agent  
- **Achats** :  
  - Transformateur4AcheteurBourse.java  
  - Transformateur4AcheteurContratCadre.java  
- **Ventes** :  
  - Transformateur4VendeurAppelDOffre.java  
  - Transformateur4VendeurContratCadre.java  
- **Transformation** :  
  - Transformation.java  
  - Transformation2.java  

## 4. Compétences mobilisées

- Programmation orientée objet (Java)  
- Simulation multi-acteurs  
- Conception de stratégies d’achat/vente  
- Négociation économique  
- Gestion de production et d’optimisation  
- Modélisation de marchés et interactions contractuelles  

## 5. Exécution

Le projet s’exécute dans le simulateur fourni dans le cadre du module EQ7.  
Les classes implémentées sont intégrées dans le moteur de simulation, qui orchestre automatiquement les interactions entre les différents acteurs du marché.



