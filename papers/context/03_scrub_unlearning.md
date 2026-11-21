# CONTEXTE AGENT — Towards Unbounded Machine Unlearning / SCRUB (Kurmanji, Triantafillou, Hayes, Triantafillou — Warwick + Google DeepMind, NeurIPS 2023, arXiv 2302.09880)
Source vérifiée : texte complet lu (corps + pseudo-code + hyperparams en annexe). Code : https://github.com/Meghdad92/SCRUB

## En une phrase
SCRUB fait « oublier » un forget set Df à un réseau déjà entraîné via un schéma **teacher-student** : l'élève (initialisé = modèle original) reste proche du prof sur le retain set Dr et s'en éloigne sur Df ; **SCRUB+R** (rewind) évite un oubli « trop fort » qui trahirait les données à une attaque par inférence d'appartenance (MIA).

## Problème : 3 applications, 3 définitions de l'oubli
| App | But | Métrique d'oubli |
|---|---|---|
| **RB** Removing Biases | erreur maximale sur Df | forget error ↑ (+ retain/test error inchangées) |
| **RC** Resolving Confusion | supprimer la confusion due à des labels faux (Df = exemples mal étiquetés) | IC-ERR (classe confuse prédite comme une autre classe) ↓, FGT-ERR (confusion 0↔1) ↓ |
| **UP** User Privacy (droit à l'oubli) | erreur sur Df ≈ celle du retrain, pas plus | MIA accuracy ≈ 50 % |
Toujours : préserver l'utilité (accuracy sur Dr et test) + être bien plus rapide que le retrain.

## Méthode SCRUB
- `d(x; wu) = KL( p(f(x; wo)) ‖ p(f(x; wu)) )` (softmax prof figé wo vs élève wu).
- Objectif : `min_wu  α/Nr Σ_{Dr} d(xr; wu) + γ/Nr Σ_{Dr} CE(f(xr; wu), yr) − 1/Nf Σ_{Df} d(xf; wu)`.
- Optimisation alternée façon GAN (sinon ça oscille) :
  - **max-step** = 1 epoch sur Df : `wu ← wu + ε ∇ mean_b d(xf; wu)` (montée de gradient sur la KL).
  - **min-step** = 1 epoch sur Dr : `wu ← wu − ε ∇ mean_b [α d(xr; wu) + γ CE]`.
  - Boucle : tant que i < MAX-STEPS → max-epoch puis min-epoch ; ensuite des min-epochs supplémentaires seuls jusqu'à STEPS (restaure l'utilité). Stop quand la forget error a monté sans dégrader la retain error.
- Hyperparams : lr initial 5e-4, **décroissance ×0.1 après quelques steps (crucial contre les oscillations)** ; Adam (petite échelle) ou SGD/Adam (grande échelle) ; weight decay 0.1 (petit) / 5e-4 (grand) ; momentum 0.9. Batch sizes séparés pour Df et Dr pour régler le ratio d'itérations. Ex. ResNet CIFAR-10 class : bs forget 512, bs retain 128, max steps 2, min steps 3 ; CIFAR-10 selective : 16/64, 5/5.
- Trop peu de max-steps → pas d'oubli ; que des max-steps → l'utilité s'effondre.

## SCRUB+R (rewinding, pour UP)
1. Construire un set de validation de même distribution que Df (ex. Df = classe 0 → validation = classe 0 seulement).
2. Entraîner SCRUB en sauvegardant un checkpoint par epoch.
3. Au dernier checkpoint (oubli max), mesurer l'erreur sur ce set de validation = erreur de référence (≈ celle d'un modèle qui n'a jamais vu Df).
4. Revenir au checkpoint dont la forget error est la plus proche de cette référence.
Utile surtout en **selective unlearning** avec un petit Df (le reste de la classe est dans Dr → bonne généralisation → besoin de rewind).

## Évaluation MIA
- **Basic MIA** : classifieur binaire entraîné sur les **losses** du modèle désappris sur Df (« in ») vs test (« out ») ; évalué sur des losses held-out équilibrées. 50 % = défense parfaite.
- **LiRA-for-unlearning** (nouvelle) : shadow models + « shadow unlearned models » (l'attaquant connaît l'algo d'unlearning et l'applique sur beaucoup de forget sets) ; pour chaque exemple cible, 2 gaussiennes de confiances (oublié vs jamais vu) → test de vraisemblance. SCRUB+R résiste le mieux.

## Baselines
Retrain (oracle), Original, Finetune (sur Dr, 10 epochs, lr 0.01), **NegGrad+** (`β·CE(Dr) − (1−β)·CE(Df)`, β=0.95 petit / 0.9999 grand — baseline forte, à toujours reporter), CF-k (geler les k premières couches + finetune), EU-k (geler + réentraîner les dernières), Fisher Forgetting, NTK Forgetting (ne scale pas), Bad-T (« mauvais prof »).

## Setup
- CIFAR-10 et Lacuna-10 (visages VGG-Faces), ResNet-18 et All-CNN, pré-entraînés sur CIFAR-100 / Lacuna-100, 3 seeds, pas d'augmentation, images 32×32.
- Petite échelle (CIFAR-5/Lacuna-5, 100 train/classe, Df = 25 ex. de la classe 0) ; grande échelle : **class unlearning** (toute la classe 5) et **selective** (100 ex. de la classe 5).
- RC : 50 ex. classe 0 étiquetés 1 et 50 ex. classe 1 étiquetés 0 (CIFAR-5).

## Résultats clés
- RC (CIFAR-5, ResNet) : IC test error SCRUB 19.0 vs Retrain 24.0, Original 56.0, NegGrad+ 47.5 ; Fgt test error SCRUB 19.7 vs Retrain 18.3, Original 92.0.
- Basic MIA (CIFAR-10, ResNet, accuracy ↓ meilleur, idéal 50) : Retrain 49.3 / 54.0 (class / selective) ; Original 71.1 / 65.3 ; SCRUB 52.2 / **78.0** (trop oublié → détectable) ; **SCRUB+R 52.2 / 58.7**.
- RB : SCRUB = top performer le plus fréquent ; très rapide (second derrière Bad-T qui oublie mal). Fisher plus lent que le retrain.

## Limites / angles hackathon
- Pas de garantie théorique ; évalué uniquement en classification d'images petite échelle ; LLM / Transformers = travail futur.
- Ironie à exploiter : oublier *trop* trahit le membership → l'oubli doit être calibré.
- Idées : démo d'unlearning (classe / visage / échantillons) + vérificateur MIA ; attaque « l'oubli est-il réel ? » (MIA, relearning, comparaison avec le retrain) ; unlearning sur petit LLM (ex. Llama) en adaptant la KL sur les distributions de tokens ; **combo avec Model Equality Testing** : test statistique « le modèle désappris ≈ retrain ? ».

## Pseudo-code minimal (PyTorch)
```python
student = copy.deepcopy(teacher); teacher.eval()
opt = torch.optim.Adam(student.parameters(), lr=5e-4)
kl = lambda s, t, T=4.0: F.kl_div(F.log_softmax(s/T, 1), F.softmax(t/T, 1), reduction="batchmean")
for epoch in range(STEPS):
    if epoch < MAX_STEPS:                       # max-step: s'éloigner du prof sur Df
        for x, _ in forget_loader:
            with torch.no_grad(): t = teacher(x)
            loss = -kl(student(x), t)
            opt.zero_grad(); loss.backward(); opt.step()
    for x, y in retain_loader:                  # min-step: rester proche sur Dr + CE
        with torch.no_grad(): t = teacher(x)
        s = student(x)
        loss = alpha * kl(s, t) + gamma * F.cross_entropy(s, y)
        opt.zero_grad(); loss.backward(); opt.step()
    save_checkpoint(student, epoch)             # pour SCRUB+R
# Valeurs vérifiées dans le repo officiel (large_scale_unlearning.ipynb) : kd_T=4 (small_scale : 2), alpha=0.001 (poids KL),
# gamma=0.99 (poids CE), lr=5e-4, msteps=2, sgda_epochs=3, momentum 0.9, wd 5e-4, del_batch_size=32, batch retain 128.
# KL du repo (DistillKL) : kl_div(log_softmax(s/T), softmax(t/T), sum) * T² / batch.
# Smoke test fonctionnel : repos/_smoke/scrub_smoke.py
# MIA basique: losses par exemple sur Df vs test -> LogisticRegression, CV -> accuracy (≈0.5 souhaité)
```

## Retour d'expérience local (smoke test `repos/_smoke/scrub_smoke.py`)
- Le max-step (montée de gradient sur la KL) est **non borné** : avec un petit batch forget (32) + SGD momentum, -KL passe de -131 à -2.5e15 → NaN. Avec le batch forget du papier (512) c'est stable. ⇒ garder peu de max-steps et de gros batches forget.
- Vérifier que le modèle original connaît bien la classe à oublier avant de mesurer l'oubli.
- Résultat (classe 1 CIFAR-10) : forget 0.974→0.000, retain 0.900→0.996, MIA basique 0.658→0.985 (sur-oubli détectable, comme prévu par le papier → SCRUB+R).
