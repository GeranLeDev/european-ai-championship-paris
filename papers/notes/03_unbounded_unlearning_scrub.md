# Analyse technique : Towards Unbounded Machine Unlearning (SCRUB)

Source : arXiv 2302.09880 (v3, 30 oct. 2023), Kurmanji, Triantafillou, Hayes, Triantafillou. NeurIPS 2023.
Texte source : `/Users/matisgeran/Pro/Hackathon/European_AI_Championship_Paris/papers/txt/unbounded_unlearning_scrub.txt`

Convention : tout ce qui n'est pas explicitement dans le papier est marqué "(inférence)".

---

## TL;DR

1. SCRUB traite l'unlearning comme un jeu teacher-student : le modèle original est le "teacher" gelé, le "student" est initialisé à partir du teacher, puis s'éloigne du teacher sur le forget set et reste proche sur le retain set, en alternant des epochs "max" (forget) et "min" (retain).
2. Sur trois applications (RB, RC, UP), SCRUB est le seul méthode jugée constamment dans le top sur la qualité d'oubli tout en préservant l'utilité, et il est bien plus rapide que le retrain (facteur de speed-up de 2 à 8 selon la configuration, Table 6).
3. SCRUB+R ajoute une procédure de rewinding (retour à un checkpoint) pour que l'erreur sur le forget set reste "juste assez haute" et défende les attaques MIA ; le papier ajoute aussi une adaptation de LiRA au cadre unlearning.

---

## 1. Problème

### 1.1 Cadre formel

- Dataset D = {(x_i, y_i)}, N exemples. Modèle f(·; w), entraîné par minimisation de la cross-entropy moyenne (Section 2).
- Modèle original w_o, entraîné sur D.
- Forget set D_f (N_f exemples), retain set D_r (N_r exemples), avec D_f ∪ D_r = D (cas standard, complément).
- Objectif : produire w_u tel que f(·; w_u) "a oublié" D_f sans dégrader l'utilité (performance sur D_r et généralisation sur un test set D_t).
- Contraintes : trois critères à satisfaire simultanément : (i) forget quality, (ii) utilité (accuracy retain et test), (iii) efficacité/scalabilité.

### 1.2 Trois applications, trois définitions de "oubli"

Le papier argumente que la définition de l'oubli dépend de l'application (contribution principale).

| Application | Définition de l'oubli | Métrique principale | Sens souhaité |
|---|---|---|---|
| **RB** (Removing Biases) | Ne plus prédire les labels associés à D_f (biais, information périmée) | Erreur sur D_f (Forget error) | Maximiser |
| **RC** (Resolving Confusion) | Résoudre la confusion entre deux classes causée par des labels erronés placés dans D_f | IC-ERR et FGT-ERR (test et retain) | Minimiser |
| **UP** (User Privacy, droit à l'oubli) | L'erreur sur D_f ne doit pas être plus élevée que celle d'un retrain : sinon MIA | Précision d'un attaquant MIA (idéal 50 %) | Viser 50 % |

Définitions exactes (Section 10 du papier) :

- Erreur : Err(D) = 1 - (1/|D|) Σ 1[argmax f(x_i; w) == y_i] (Éq. 5).
- IC-ERR(D, A, B; w) = (Σ_{k≠A} C_{A,k} + Σ_{k'≠B} C_{B,k'}) / (|D_A| + |D_B|) (Éq. 6). C est la matrice de confusion ; le numérateur compte les erreurs des exemples des classes A ou B, vers n'importe quelle autre classe.
- FGT-ERR(D, A, B; w) = C_{A,B} + C_{B,A} (Éq. 7). Ne compte que la confusion entre A et B.

Métriques utilisées par application :
- **RB** : Forget error (maximiser), tout en gardant Retain error et Test error bas.
- **RC** : IC test/retain error et Fgt test/retain error (minimiser), plus les erreurs standard.
- **UP** : Basic MIA et LiRA-for-unlearning (la précision de l'attaquant doit tendre vers 50 %, ou AUC faible).

---

## 2. Méthode SCRUB en détail

### 2.1 Notations et divergence

Teacher = w_o (gelé). Student = w_u (optimisé).

d(x; w_u) = D_KL( p(f(x; w_o)) || p(f(x; w_u)) )

Il s'agit de la KL entre les distributions softmax du teacher et du student sur l'exemple x.

### 2.2 Idée

- Initialisation : w_u ← w_o. Le student est déjà bon sur D_r et D_f au départ, car le teacher a été entraîné sur tout D.
- Sur D_r : rester proche du teacher (minimiser d).
- Sur D_f : s'éloigner du teacher (maximiser d).

### 2.3 Objectifs successifs

- Éq. 1 (naïf, uniquement forget) : min_w_u  - (1/N_f) Σ_{x_f ∈ D_f} d(x_f; w_u). Les auteurs observent que cela dégrade aussi D_r.
- Éq. 2 (contrastif) : min_w_u (1/N_r) Σ_{x_r ∈ D_r} d(x_r; w_u) - (1/N_f) Σ_{x_f ∈ D_f} d(x_f; w_u).
- Éq. 3 (objectif final, ajoute la tâche sur D_r) :

  min_w_u  (α/N_r) Σ_{x_r ∈ D_r} d(x_r; w_u)  +  (γ/N_r) Σ_{(x_r, y_r) ∈ D_r} l(f(x_r; w_u), y_r)  -  (1/N_f) Σ_{x_f ∈ D_f} d(x_f; w_u)

  où l est la cross-entropy, et α, γ sont des hyperparamètres scalaires.

### 2.4 Problème d'optimisation et recette alternée

Les auteurs disent que l'optimisation directe de l'Éq. 3 oscille (deux objectifs qui interfèrent). La solution : alterner des "max-steps" (sur D_f) et des "min-steps" (sur D_r), à la manière des GAN (discriminateur entraîné plusieurs fois avant chaque update du générateur).

Algorithme 1 (SCRUB, tel qu'écrit dans l'Appendice, Section 9) :

```
Require: poids teacher w_o, MAX-STEPS, STEPS, taux d'apprentissage ε
w_u ← w_o
i ← 0
repeat
    if i < MAX-STEPS then
        w_u ← DO-MAX-EPOCH(w_u)      # sur D_f
    end if
    w_u ← DO-MIN-EPOCH(w_u)          # sur D_r
until i < STEPS
```

Note : la pseudo-code du papier ne montre pas l'incrément de i (inférence : c'est probablement un oubli de typographie, à implémenter avec i += 1 à chaque itération).

Séquence complète (Section 3.1) : alternance (max-epoch, min-epoch) pendant MAX-STEPS, puis des min-epochs supplémentaires pour restaurer l'utilité sur D_r. L'arrêt se fait quand l'erreur sur D_f a augmenté sans dégrader l'erreur sur D_r. Les auteurs disent qu'un petit nombre d'epochs suffit en pratique.

Algorithme 2 (DO-MAX-EPOCH) : batches de D_f de taille B_f, pas de gradient ASCENDANT sur la moyenne de d :

```
for batch b in D_f (taille B_f):
    w_u ← w_u + ε * ∇_{w_u} (1/|b|) Σ_{x ∈ b} d(x; w_u)
```

Algorithme 3 (DO-MIN-EPOCH) : batches de D_r de taille B_r, pas de gradient DESCENDANT sur α·d + γ·CE :

```
for batch b in D_r (taille B_r):
    w_u ← w_u - ε * ∇_{w_u} (1/|b|) Σ_{(x,y) ∈ b} [ α·d(x; w_u) + γ·l(f(x; w_u), y) ]
```

### 2.5 Rôle de la cross-entropy (Figure 10)

Le terme γ·CE sur D_r est un ajout "petit mais consistant" qui protège l'utilité (Section 12 et Figure 10). Au prix parfois d'une erreur plus faible sur D_f.

### 2.6 Hyperparamètres (Tables 3, Section 9)

- Optimiseur : Adam en small-scale (meilleur selon les auteurs), Adam ou SGD en large-scale. Même optimiseur pour min et max.
- Taux d'apprentissage initial : 0.0005, décroissance ×0.1 après un certain nombre de min et max steps.
- Weight decay : 0.1 (small-scale), 0.0005 (large-scale). Momentum : 0.9.
- Tailles de batch différentes entre D_f et D_r, pour contrôler le nombre d'itérations dans chaque sens.
- α et γ : hyperparamètres, tunés pour obtenir une forte erreur sur D_f tout en conservant l'erreur retain et validation de l'original. Figure 5 : SCRUB est peu sensible à α et γ (grille {0.1, 0.5, 1, 2, 3}).

Table 3 (valeurs telles que publiées ; colonnes : forget bs, retain bs, max steps, min steps) :

| Modèle | Dataset | Type d'unlearning | Forget bs | Retain bs | Max steps | Min steps |
|---|---|---|---|---|---|---|
| ResNet | CIFAR-10 | class | 512 | 128 | 2 | 3 |
| ResNet | CIFAR-10 | selective | 16 | 64 | 5 | 5 |
| ResNet | Lacuna-10 | class | 128 | 128 | 5 | 5 |
| ResNet | Lacuna-10 | selective | 32 | 32 | 4 | 4 |
| ResNet | CIFAR-5 | selective | 32 | 32 | 10 | 10 |
| ResNet | Lacuna-5 | selective | 32 | 32 | 5 | 10 |
| All-CNN | CIFAR-10 | class | 512 | 256 | 3 | 4 |
| All-CNN | CIFAR-10 | selective | 16 | 64 | 5 | 5 |
| All-CNN | Lacuna-10 | class | 32 | 32 | 4 | 4 |
| All-CNN | Lacuna-10 | selective | 8 | 32 | 2 | 4 |
| All-CNN | CIFAR-5 | selective | 16 | 32 | 5 | 10 |
| All-CNN | Lacuna-5 | selective | 32 | 32 | 5 | 10 |

Autres points :
- Table 3 : les hyperparamètres ci-dessus sont pour SCRUB ; ils sont choisis par expérience, pas par théorie.
- Pas de data augmentation (choix de cohérence avec la littérature, Section 11 "Limitations").

### 2.7 SCRUB+R (rewinding)

Problème : SCRUB pousse l'erreur sur D_f très haut, ce qui peut rendre les exemples oubliés identifiables par MIA (Section 3.2).

Procédure (Section 3.2) :
1. Construire un **validation set** de même distribution que D_f (si D_f ne contient que des exemples de la classe 0, le validation set ne contient que des exemples de la classe 0). Le validation set est issu de données non vues par le modèle final (détail d'échantillonnage non précisé dans l'extrait, voir Section 9 : 25 exemples par classe pour les validation sets small-scale).
2. Entraîner SCRUB normalement, en sauvegardant un checkpoint à chaque epoch.
3. À la fin de SCRUB (là où l'erreur sur D_f est typiquement la plus haute), mesurer l'erreur sur le validation set : c'est la **référence**.
4. "Rewind" : choisir le checkpoint dont l'erreur sur D_f est la plus proche de cette référence.

Justification : la dernière étape approxime "oubli maximal" ; l'erreur sur un validation set de même distribution reflète alors la généralisation pure, c'est-à-dire ce qu'un modèle n'ayant jamais vu D_f produirait (Section 3.2).

Implémentation pour LiRA (inférence documentée par les auteurs, Section 11.2) : au lieu de faire le rewinding, ils filtrent les runs SCRUB dont l'erreur sur D_f n'est pas proche de l'erreur validation, avec un seuil de 0.1.

Limitation (Section 8) : parfois, l'erreur finale de SCRUB est "trop basse" (arrêt prématuré), et le rewinding ne peut pas la remonter.

### 2.8 Ablations sur la dynamique (Section 12, Figures 6-9)

- Max-steps seuls (Éq. 1) : oubli obtenu, mais dégradation de D_r et du validation set.
- Trop peu de max-steps : l'erreur sur D_f redescend.
- Sans la phase finale de min-steps : mauvais compromis.
- Bon équilibre : SCRUB (Figure 6d).

---

## 3. Évaluation de l'oubli : la MIA

### 3.1 Basic MIA (Section 11.1)

Construction :
- Entraîner un classifieur binaire (l'attaquant) sur les **losses** (cross-entropy) du modèle non-oubli w_u, pour des exemples de D_f (label 1 = "in") et de D_t (label 0 = "out").
- Dataset d'entraînement de l'attaquant : {(l(x_i, y_i), y_i^b)} ; équilibré entre D_f et D_t.
- Évaluation sur un ensemble disjoint, également équilibré.
- Attaquant utilisé : LogisticRegression de scikit-learn, valeurs de loss clippées dans [-400, +400], validation croisée à 5 folds, score = accuracy moyenne sur les parties évaluation.
- Le test set doit être de même distribution que D_f (ex. : si D_f contient des exemples de la classe "chat", D_t aussi), sinon la MIA reconnaît la distribution et non le membership.
- Score idéal pour une défense parfaite : 50 %.
- Le modèle Retrain est théoriquement optimal pour cette métrique.

Résultats Basic MIA (Table 2, CIFAR-10) :
- Retrain : 49.33 (ResNet class), 54.00 (ResNet selective), 55.00 (All-CNN class), 48.73 (All-CNN selective).
- Original : 71.10 / 65.33 / 66.50 / 71.40.
- SCRUB : 52.20 / 78.00 / 52.00 / 54.30.
- SCRUB+R : 52.20 / 58.67 / 52.00 / 54.30.
- Le rewinding n'est déclenché que pour ResNet selective (Table 2 et Section 5.4).

### 3.2 LiRA-for-unlearning (Section 11.2)

Contexte : LiRA (Carlini et al. 2022) entraîne des "shadow models" pour estimer deux gaussiennes par exemple cible (confiance sous "in" et sous "out") puis fait un test de rapport de vraisemblance.

Adaptation proposée par les auteurs (contribution du papier) :
- Pour chaque shadow model, l'attaquant produit aussi un "shadow unlearned" en appliquant l'algorithme d'unlearning plusieurs fois avec de nombreux forget sets.
- Pour chaque exemple cible, il estime deux gaussiennes : confiance quand l'exemple a été oublié vs confiance quand il n'a jamais été vu.
- Prédiction : "oublié" si la vraisemblance sous la première gaussienne est supérieure à la seconde.
- Confiance : softmax(f(x))[y] (probabilité de la bonne classe), avec logit-scaling pour rendre la distribution gaussienne (Carlini et al. 2022).

Configuration expérimentale (Section 11.2) :
- CIFAR-10, sélectif, forget set de 200 exemples de la classe 5.
- 256 "shadow original" models, entraînés sur des moitiés aléatoires de la moitié du train set.
- Pour chaque shadow original : 10 000 forget sets distincts, donc 10 000 runs d'unlearning par shadow.
- Target model : un modèle original entraîné sur une moitié, puis unlearning sur un forget set de 200 exemples de classe 5.

Métrique : courbe ROC en échelle log-log (Figure 4), AUC et TPR à faible FPR (Carlini et al. recommandent cette zone).

Résultats AUC (Figure 4, 4 cibles différentes, valeurs rapportées) :
- Original : 0.92
- SCRUB : 0.72 à 0.74
- SCRUB+R : 0.72 à 0.74
- NegGrad+ : 0.79 à 0.80
- CF-k : 0.91 à 0.92
- EU-k : 0.81 à 0.84

### 3.3 Limites de la MIA (Section 11, "Limitations")

- Les auteurs précisent que le modèle original est sur-entraîné (pas d'augmentation de données), donc sa vie privée est mauvaise ; la MIA est donc plus forte que sur un modèle moderne.
- Basic MIA : simple et très dépendante de la distribution du test set.

---

## 4. Baselines comparées

Source : Section 5.1 et Section 9.

| Méthode | Description |
|---|---|
| **Retrain** | Réentraînement from scratch sur D_r. Référence (non viable en pratique). |
| **Original** | Modèle entraîné sur D, sans unlearning. |
| **Finetune** | Fine-tuning de l'original sur D_r, 10 epochs, lr 0.01, weight decay 0.0005. |
| **NegGrad+** | Fine-tuning sur D_r et D_f avec gradient négatif sur D_f. Loss : β·(1/|D_r|)Σ l_r - (1-β)·(1/|D_f|)Σ l_f. β = 0.95 en small-scale (10 epochs, SGD, lr 0.01, wd 0.1) ; β = 0.9999 en large-scale (5 epochs, SGD, lr 0.01, wd 0.0005). Les auteurs précisent qu'un β petit fait exploser le modèle. Variante plus forte que NegGrad (Golatkar / Graves) qui n'entraîne que sur D_f. |
| **CF-k** | Catastrophic Forgetting-k (Goel et al. 2022) : geler les k premières couches, fine-tuner le reste sur D_r. Même réglage que Finetune. |
| **EU-k** | Exact Unlearning-k (Goel et al. 2022) : geler les k premières couches, réinitialiser le reste (poids pré-entraînés), réentraîner sur D_r avec les réglages de Retrain. Pour ResNet : bloc4 libre ; pour All-CNN : 9e bloc séquentiel. |
| **Fisher Forgetting** | Golatkar et al. 2020a (code public utilisé par les auteurs). Lent : dépasse le temps de retrain (Figure 1). |
| **NTK Forgetting** | Golatkar et al. 2020b (code public). Ne passe pas à l'échelle au-delà du small-scale. |
| **Bad-T** | Chundawat et al. 2022 (incompetent teacher). Réimplémenté par les auteurs (pas de code). 1 epoch d'unlearning, Adam, température T = 4, tout le retain set (au lieu de 30 % dans le papier original). |

---

## 5. Setup expérimental

### 5.1 Datasets

- **CIFAR-10** : 10 classes, images 32x32. Train 40 000, test 10 000, validation 10 000 (Section 9).
- **Lacuna-10** : dérivé de VGG-Faces (Golatkar et al. 2020a). 10 célébrités (classes) d'au moins 500 images ; 100 images par classe en test ; le reste en train. Images redimensionnées en 32x32.
- **CIFAR-100 et Lacuna-100** : utilisés pour pré-entraîner les modèles (pas de chevauchement avec les 10 classes).
- **CIFAR-5 et Lacuna-5** (small-scale) : 5 premières classes, 100 exemples par classe en train, 25 par classe en validation, 100 en test (Golatkar et al. 2020b).
- Aucune data augmentation.

### 5.2 Scénarios

- **Small-scale** : CIFAR-5 / Lacuna-5, forget set de 25 exemples de la classe 0 (5 %). Permet de comparer avec NTK (qui ne passe pas à l'échelle).
- **Large-scale, class unlearning** : forget set = tout le train set de la classe 5 (10 %).
- **Large-scale, selective unlearning** : forget set = 100 exemples de la classe 5 (0.25 % sur CIFAR, 2 % sur Lacuna).
- **RC** : mislabelling de 50 + 50 exemples dans CIFAR-5 et Lacuna-5 (Table 1, RC summary), puis 2000 + 2000 (CIFAR-10) et 200 + 200 (Lacuna-10) pour les tableaux larges.

### 5.3 Architectures

- **ResNet-18** (He et al. 2016) : pour les expériences large-scale, architecture exacte ; en small-scale, nombre de filtres réduit de 60 % dans chaque bloc.
- **All-CNN** (Springenberg et al. 2014) : couches réduites, batch normalization ajoutée avant chaque non-linéarité ; même réduction de filtres en small-scale.
- **VGG16+BN** (~138 M paramètres), CIFAR-100, forget sets variables : expérience "larger-scale" (Section 13, Tables 4 et 5).

### 5.4 Pré-entraînement

- Modèles pré-entraînés 30 epochs sur CIFAR-100 (ou Lacuna-100), SGD, lr fixe 0.1, cross-entropy, weight decay 0.0005, momentum 0.9, batch 128.
- 3 graines aléatoires par expérience, mean et std rapportés (sauf Tables 4 et 5 : 3 runs et 10 runs).

### 5.5 Compute

- Expériences de scale-up : Python 3.8, Ubuntu 20, 40 cœurs CPU, GPU Nvidia GTX 2080, 256 Go RAM (Section 9).

---

## 6. Résultats chiffrés clés

### 6.1 RB (Figures 1 et 2, Tables 7-12)

- Figure 2 (nombre de fois où chaque méthode est top-performer en forgetting, IC de 95 %) : SCRUB est la plus constante. Petit-scale : compté sur 4 cas ({ResNet, All-CNN} x {CIFAR, Lacuna}). Large-scale : 8 cas (ajoute class et selective).
- Table 7 (CIFAR-5, ResNet, small-scale) : SCRUB forget error 40.8 ± 1.8 (CIFAR-5), 24.8 ± 5.2 (Lacuna-5), tandis que retrain donne 28.8 et 4.8. Test error SCRUB 24.2 (original 24.2), donc utilité préservée.
- Table 9 (CIFAR-10 class, ResNet, large-scale) : SCRUB forget error 100.0 ± 0.0 (retrain 100.0), test error 15.73 (retrain 14.72, original 16.56).
- Table 10 (CIFAR-10 class, All-CNN, large-scale) : SCRUB forget 100.0, test 15.06 (retrain 13.97).
- Table 11 (CIFAR-10 selective, ResNet, large-scale) : SCRUB forget 70.33 ± 4.16 (retrain 29.67), test 18.04 (retrain 17.4).
- Table 12 (CIFAR-10 selective, All-CNN) : SCRUB forget 29.33 (retrain 25.67), test 16.55 (retrain 16.47). Dans ce cas SCRUB colle au retrain.

Remarque : dans RB le papier considère que plus d'erreur sur D_f est mieux, donc un écart "au-dessus du retrain" n'est pas pénalisé ici.

### 6.2 RC (Table 1 et Tables 13-20)

Table 1 (CIFAR-5, 50 + 50 mislabelled, moyennes ± std) :

| Modèle | IC test (ResNet) | IC retain (ResNet) | Fgt test (ResNet) | Fgt retain (ResNet) | IC test (All-CNN) | Fgt test (All-CNN) |
|---|---|---|---|---|---|---|
| Original | 56.0 ± 3.0 | 0.0 | 92.0 ± 7.9 | 0.0 | 49.0 ± 4.8 | 80.7 ± 12.6 |
| EU-k | 47.0 ± 4.8 | 8.3 ± 4.7 | 63.3 ± 9.7 | 3.7 ± 2.5 | 59.5 ± 5.2 | 68.7 ± 15.6 |
| Fisher | 51.5 ± 7.5 | 26.3 ± 9.5 | 79.0 ± 3.6 | 20.0 ± 7.9 | 56.8 ± 8.7 | 78.3 ± 15.5 |
| NTK | 37.5 ± 4.0 | 0.0 | 52.0 ± 10.6 | 0.0 | 36.7 ± 4.1 | 54.3 ± 9.0 |
| Bad-T | 47.3 ± 9.7 | 6.0 ± 6.2 | 27.0 ± 8.7 | 0.3 ± 0.6 | 39.3 ± 8.2 | 25.7 ± 5.0 |
| **SCRUB** | **19.0 ± 3.9** | **0.0** | **19.7 ± 7.5** | **0.0** | **26.0 ± 4.4** | **18.0 ± 11.1** |

Autres tableaux RC (valeurs extraites, "Fgt test" = erreur de confusion entre les deux classes sur test) :
- Table 16 (Lacuna-5, All-CNN) : SCRUB test 3.87, IC test 4.33, Fgt test 0.0. Retrain : test 4.2, IC test 5.33, Fgt test 0.0.
- Table 15 (Lacuna-5, ResNet) : SCRUB test 8.47, IC test 11.33 ± 3.82, Fgt test 9.33 ± 1.53. Retrain : test 6.0, IC test 7.17, Fgt test 0.0.
- Table 17 (CIFAR-10 ResNet, 2000 + 2000) : SCRUB test 18.01, IC test 15.07, Fgt test 30.33. Retrain : test 18.7, IC test 14.78, Fgt test 31.33.
- Table 19 (Lacuna-10 ResNet, 200 + 200) : SCRUB test 3.07, IC test 6.83, Fgt test 0.67. Retrain : test 2.43, IC test 3.33, Fgt test 0.0.

Synthèse de l'auteur (Section 5.3) : SCRUB est "by far the most consistent".

### 6.3 UP (Tables 2, 21-28, Figures 3-4)

- Table 2 (CIFAR-10 basic MIA) : SCRUB+R 58.67 pour ResNet selective (Retrain 54.00, Original 65.33) ; SCRUB 78.00 pour la même configuration (ratée). Rewinding déclenché uniquement là.
- Table 21 (CIFAR-10 All-CNN selective) : SCRUB MIA 51.00, SCRUB+R 49.33, Retrain 51.33, Original 68.67, NegGrad+ 73.00, Bad-T 77.67.
- Table 22 (CIFAR-10 All-CNN class) : SCRUB MIA 54.30, Retrain 48.73, Original 71.40, NegGrad+ 70.03.
- Table 23 (CIFAR-10 ResNet selective) : SCRUB 78.00, SCRUB+R 58.67, Retrain 54.00, Original 65.33, Bad-T 54.33, EU-k 54.20.
- Table 24 (CIFAR-10 ResNet class) : SCRUB 52.20, SCRUB+R 52.20, Retrain 49.33, Original 71.10, EU-k 54.20.
- Figure 3 (CIFAR-10 ResNet, forget sets de 100, 150, 175, 200 exemples) : rewinding plus utile pour les petits forget sets. SCRUB+R défend "comparablement à Retrain" (valeurs exactes non extraites du texte).
- Figure 4 (LiRA-for-unlearning, AUC) : voir section 3.2.

### 6.4 Grands modèles (Tables 4, 5 et 6)

Table 4 (VGG16+BN, CIFAR-100, class-0 = 400 exemples, 3 runs) :
- Original : test 35.84, forget 0.00, MIA 61.67.
- Retrain : test 36.92, forget 100.00, MIA 55.67.
- SCRUB : test 38.27 ± 3.08, retain 0.78, forget 100.00, MIA 48.33.
- SCRUB+R : test 37.60, forget 100.00, MIA 57.33.
- NegGrad+ : test 43.16, retain 6.86, forget 49.50, MIA 64.67.
- Bad-T@epoch10 : test 43.33, forget 85.33, MIA 63.33.

Table 5 (VGG16+BN, CIFAR-100, selective, 10 exemples de la classe 0, 10 runs) :
- Retrain : test 37.54, forget 19.00, MIA 45.00.
- SCRUB : test 38.18, retain 0.48, forget 45.00, MIA 56.00.
- SCRUB+R : test 38.40, retain 0.69, forget 40.00, MIA 61.00.
- NegGrad+ : test 45.98, forget 44.00, MIA 55.00.

Table 5 (même dataset, 20 exemples, selon le texte extrait) : SCRUB+R MIA 49.50, forget 23.00 ; Retrain MIA 49.00, forget 19.50.

Table 6 (speed-up, runtime retrain / runtime unlearning) :
- SCRUB : 7.84 (CIFAR-10 ResNet class), 7.41 (ResNet selective), 6.36 (All-CNN class), 5.33 (All-CNN selective), 2.17 / 1.95 / 2.81 / 2.48 (Lacuna-10).
- Bad-T : 19.07 / 20.44 / 17.91 / 17.03 (CIFAR-10), 20.05 / 20.27 / 16.32 / 16.02 (Lacuna-10). Le papier précise que Bad-T est plus rapide mais ne fonctionne pas.
- EU-k : environ 1.2 à 1.4.
- Fisher : environ 0.07 à 0.16 (plus lent que retrain).

### 6.5 Dynamique de la défense LiRA (Section 11.2)

Voir Figure 4 (section 3.2). SCRUB(+R) a la meilleure défense, surtout dans la zone de faible FPR.

---

## 7. Limites

Selon la Section 8 (Limitations and Future Work), la Section 11 et la Section 6 :

1. **Pas de garanties théoriques** : les auteurs le disent explicitement. Le papier ne prouve pas que SCRUB efface l'influence de D_f.
2. **Instabilité du min-max** : oscillations si la séquence max/min est mal réglée (Figure 6). Décroissance du lr nécessaire.
3. **Dépendance aux hyperparamètres** : α, γ, nombre de max steps, nombre de min steps, tailles de batch. Tunés par expérience (Tables 3 et 5). Le papier note une faible sensibilité à α, γ (Figure 5), mais pas aux autres.
4. **Rewinding** : peut ne pas suffire si l'erreur forget finale est trop basse (arrêt prématuré). Dans l'autre sens (trop haute), il aide.
5. **Échelle** : datasets et modèles restent petits (CIFAR-10, Lacuna-10, ResNet-18, All-CNN, plus VGG16 sur CIFAR-100). Pas testé sur des transformers, du self-supervised learning ou des modèles de langage.
6. **Pas d'augmentation de données** : le modèle original est sur-entraîné et a une vie privée faible (ex. : TPR élevé sur Figure 4). Résultats MIA non directement transposables à un modèle state-of-the-art (limite assumée par les auteurs).
7. **Bad-T** : réimplémenté par les auteurs (pas de code public disponible au moment du papier), ce qui peut biaiser la comparaison (inférence).
8. **Basic MIA** : simple, dépendante de la distribution du test set (Section 11.1).
9. **Définition de l'oubli** : les auteurs reconnaissent qu'il n'existe pas de définition formelle et standard de l'unlearning (Section 6).
10. **Broader impact** : SCRUB est un post-traitement qui ne garantit pas qu'un biais ou un problème de vie privée est retiré ; les auteurs appellent à un audit des modèles (Section 6).

---

## 8. Code publié

- Code de reproduction annoncé dans le texte, section 3.1, note de bas de page 2 : **https://github.com/Meghdad92/SCRUB**
- Le papier cite aussi la compétition Kaggle "NeurIPS 2023 - Machine Unlearning" (référence Triantafillou et al. 2023) : **https://kaggle.com/competitions/neurips-2023-machine-unlearning** (URL dans la bibliographie).
- Code des baselines NTK et Fisher : publics (Golatkar et al.), utilisés par les auteurs (Section 5.1). URLs non données dans le texte.
- Bad-T : pas de code public, réimplémenté par les auteurs.

---

## 9. Idées de challenges hackathon et pistes concrètes d'implémentation

### 9.1 Ce qu'on peut faire en un hackathon (inférence)

Objectif réaliste : implémenter SCRUB sur un petit modèle (ResNet-18 ou petit CNN) sur CIFAR-10 ou un sous-ensemble (CIFAR-5), mesurer forget error, retain error, test error et une MIA basique, puis comparer à Retrain, Finetune, NegGrad+.

Challenges possibles :

- **Challenge RB** : maximiser l'erreur sur un ensemble de classe "biaisée" tout en conservant test error à moins de 1-2 points de l'original. Métrique : forget error (à maximiser) et test error (à minimiser).
- **Challenge RC** : des labels bruités entre deux classes ; l'unlearning doit éliminer la confusion. Métrique : FGT-ERR (Éq. 7) sur test.
- **Challenge UP** : après unlearning, un attaquant MIA doit échouer (accuracy proche de 50 %). Métrique : Basic MIA avec LogisticRegression sur les losses. Bonus : SCRUB+R avec un validation set de même distribution.
- **Challenge combiné** : une seule méthode doit être bonne sur les trois métriques, scorée par un mix pondéré (inférence sur le design du scoring).

### 9.2 Pseudo-code SCRUB (PyTorch-like, inférence fidèle au papier)

```python
import copy, torch
import torch.nn.functional as F

def kl_teacher_student(logits_teacher, logits_student):
    # d(x; w_u) = KL( p(f(x; w_o)) || p(f(x; w_u)) ), par exemple
    p_t = F.softmax(logits_teacher, dim=1)
    log_p_s = F.log_softmax(logits_student, dim=1)
    return (p_t * (torch.log(p_t + 1e-12) - log_p_s)).sum(dim=1)  # shape [B]

def scrub_unlearn(model_orig, forget_loader, retain_loader,
                  max_steps, min_steps, alpha, gamma,
                  lr=5e-4, weight_decay=0.1, momentum=0.9,
                  optimizer_name="adam", lr_decay_every=None):
    teacher = copy.deepcopy(model_orig).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = copy.deepcopy(model_orig).train()
    opt = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay) \
        if optimizer_name == "adam" else \
        torch.optim.SGD(student.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    sched = None
    if lr_decay_every:
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_decay_every, gamma=0.1)

    def max_epoch():
        # Algorithme 2 : ascension de gradient sur d moyen, forget set
        for xf, _ in forget_loader:
            with torch.no_grad():
                t_logits = teacher(xf)
            s_logits = student(xf)
            loss_max = -kl_teacher_student(t_logits, s_logits).mean()  # on minimise -d pour monter d
            opt.zero_grad(); loss_max.backward(); opt.step()

    def min_epoch():
        # Algorithme 3 : descente sur alpha*d + gamma*CE, retain set
        for xr, yr in retain_loader:
            with torch.no_grad():
                t_logits = teacher(xr)
            s_logits = student(xr)
            d = kl_teacher_student(t_logits, s_logits)
            ce = F.cross_entropy(s_logits, yr, reduction="none")
            loss_min = (alpha * d + gamma * ce).mean()
            opt.zero_grad(); loss_min.backward(); opt.step()

    # Boucle principale (Algorithme 1)
    i = 0
    while True:
        if i < max_steps:
            max_epoch()
        min_epoch()
        i += 1                      # inférence : l'incrément manque dans la pseudo-code du papier
        if sched: sched.step()
        if i >= max_steps and i >= min_steps:  # inférence : arrêt simplifié
            break

    # Phase finale de min-steps (Section 3.1) : restaure l'utilité sur D_r
    for _ in range(min_steps):
        min_epoch()

    return student
```

Notes d'implémentation (inférence) :
- Le papier ne précise pas de température pour la KL dans SCRUB (contrairement à Bad-T, T = 4).
- Le critère d'arrêt exact n'est pas formalisé dans le papier : "arrêt quand l'erreur forget a augmenté sans dégrader l'erreur retain". Le code de référence (Meghdad92/SCRUB) est la source de vérité.
- Le "max_steps" et "min_steps" de Table 3 sont des nombres d'epochs.

### 9.3 Pseudo-code SCRUB+R (rewinding, inférence)

```python
def scrub_with_rewind(model_orig, forget_loader, retain_loader, val_forget_loader,
                      max_steps, min_steps, alpha, gamma, **kw):
    checkpoints = []  # un checkpoint par epoch (Section 3.2)
    student = copy.deepcopy(model_orig)
    # ... boucle SCRUB identique, mais après chaque epoch :
    #     checkpoints.append(copy.deepcopy(student.state_dict()))

    # Référence : erreur du modèle final sur un validation set de même distribution que D_f
    ref_err = error_rate(final_student, val_forget_loader)

    # Rewind : checkpoint dont l'erreur sur D_f est la plus proche de ref_err
    best = min(checkpoints, key=lambda c: abs(error_rate(load(c), forget_loader) - ref_err))
    return load(best)
```

### 9.4 MIA basique (inférence fidèle à Section 11.1)

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def per_example_loss(model, loader):
    # cross-entropy par exemple, sans réduction
    losses = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            losses.append(F.cross_entropy(model(x), y, reduction="none").cpu().numpy())
    return np.concatenate(losses)

def basic_mia(model, forget_loader, test_loader_same_dist):
    l_f = np.clip(per_example_loss(model, forget_loader), -400, 400)  # clipping du papier
    l_t = np.clip(per_example_loss(model, test_loader_same_dist), -400, 400)
    X = np.concatenate([l_f, l_t]).reshape(-1, 1)
    y = np.concatenate([np.ones_like(l_f), np.zeros_like(l_t)])     # 1 = forget ("in")
    clf = LogisticRegression()
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")  # 5 splits, comme le papier
    return scores.mean()   # idéal : ~0.5
```

Points d'attention (inférence) :
- Le papier équilibre les deux classes et utilise un ensemble d'évaluation disjoint du train de l'attaquant. Une validation croisée sur le même vecteur suffit pour un hackathon, mais attention à la fuite de données si les mêmes exemples apparaissent dans les folds.
- Le test set doit être de la même distribution que le forget set (ex. : même classe), sinon la MIA mesure la distribution et non le membership.

### 9.5 Pistes de scope (inférence)

- Plus simple : reproduire un seul cas (CIFAR-5 small-scale ResNet, RB ou RC), avec Retrain comme référence.
- Plus ambitieux : SCRUB + rewind + Basic MIA, comparé à NegGrad+ et Finetune sur CIFAR-10.
- LiRA complet est très coûteux (256 shadow models, 10 000 unlearning runs chacun dans le papier, Section 11.2). Pour un hackathon, un LiRA réduit (quelques dizaines de shadow models) est une approximation, à signaler comme telle.

---

## 10. Glossaire

- **Unlearning (désapprentissage)** : retirer de l'influence d'un sous-ensemble d'entraînement d'un modèle entraîné, sans retrain complet.
- **Forget set (D_f)** : exemples à oublier.
- **Retain set (D_r)** : exemples que le modèle doit conserver.
- **Test set (D_t)** : exemples non vus, pour mesurer la généralisation.
- **Teacher / Student** : modèle original gelé / modèle en cours d'unlearning, initialisé depuis le teacher.
- **Max-step / min-step** : epoch d'ascension sur D_f (rendre le student différent du teacher) / epoch de descente sur D_r (rester proche du teacher et apprendre la tâche).
- **Divergence KL** : mesure d'écart entre deux distributions de probabilités, ici entre les sorties softmax du teacher et du student.
- **Cross-entropy (CE)** : loss de classification standard.
- **RB, RC, UP** : Removing Biases, Resolving Confusion, User Privacy.
- **IC-ERR / FGT-ERR** : erreurs de confusion inter-classes (Éq. 6 et 7).
- **MIA (Membership Inference Attack)** : attaque qui devine si un exemple a été utilisé pour l'entraînement, en observant le modèle.
- **Basic MIA** : attaquant simple, régression logistique sur les losses.
- **LiRA** : Likelihood Ratio Attack (Carlini et al. 2022), attaque par rapport de vraisemblance avec shadow models.
- **Shadow models** : modèles auxiliaires entraînés par l'attaquant, pour estimer la distribution des confiances.
- **Rewinding (SCRUB+R)** : choix d'un checkpoint intermédiaire de SCRUB dont l'erreur sur D_f est proche d'une référence validation.
- **Retrain (oracle)** : modèle réentraîné from scratch sur D_r, référence d'unlearning exact.
- **Speed-up / scale-up factor** : ratio entre runtime de retrain et runtime de l'unlearning (Table 6).
- **NegGrad+** : fine-tuning avec gradient négatif sur D_f, pondéré par β.
- **CF-k / EU-k** : fine-tuning / retraining des couches au-delà des k premières, avec les k premières gelées.
- **Bad-T** : unlearning par "mauvais teacher" (Chundawat et al. 2022).
- **NTK, Fisher** : méthodes d'unlearning approximatives de Golatkar et al. (2020a, b).
- **Lacuna-10** : dataset de visages de célébrités dérivé de VGG-Faces (Golatkar et al. 2020a).
- **Small-scale / large-scale** : CIFAR-5 et Lacuna-5 (petit) / CIFAR-10 et Lacuna-10 (grand), plus VGG16 sur CIFAR-100 en Section 13.
