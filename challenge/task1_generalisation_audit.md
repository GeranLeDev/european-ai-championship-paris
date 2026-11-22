# Task 1 (localisation de watermark) — audit de généralisation public → caché

Dim. 22 nov. 2026, 10:10. Question posée : notre score public (30 % du test, 306 images) est-il sur-ajusté, et que peut-on attendre sur les 70 % cachés (714 images) ?

**Réponse courte** : rien n'indique un sur-ajustement au score public. Les six fichiers soumis ont tous obtenu en public le score mesuré avant l'envoi en validation locale, à 0,008 près au pire. Sur les 70 % cachés, on attend le score local (0,822 pour le fichier actuel) à ± 0,02 près, par le seul hasard du découpage.

**Limite de cet audit** : le fichier actuellement en tête (`submission_t1_samsnap_VWY.pt`, 0,82 public) est sur le Mac d'Amil, pas dans ce dépôt. Le contrôle de réalisme (§3) a été fait sur les fichiers présents ici ; la commande pour le refaire sur le fichier final est au §5.

## 1. Le score local a prédit le score public à chaque envoi

| Fichier | Score local avant envoi | Score public | Écart |
|---|---|---|---|
| `submission_baseline.pt` | 0,466 | 0,4584 | −0,008 |
| Modal, envoi de 00:17 | 0,750 | 0,7523 | +0,002 |
| `submission_t1_v3a.pt` | 0,767 | 0,7656 | −0,001 |
| `submission_t1_v4c.pt` | 0,784 | 0,7809 | −0,003 |
| `submission_t1_v5c.pt` | 0,801 | 0,7968 | −0,004 |
| `submission_t1_samsnap_VWY.pt` | 0,822 | 0,82 (2 décimales connues ici) | ≈ −0,002 |

Source : messages horodatés de `challenge/COORD_T1.md` (chaque score local y est annoncé avant l'envoi).

- Écart moyen : −0,003. Écart maximal : 0,008. Le bruit attendu du découpage seul est de ± 0,021 (§2).
- Un fichier réglé sur le score public aurait un score public supérieur au score local. C'est l'inverse qu'on observe (léger déficit, non significatif).
- Chaque gain local s'est retrouvé en public avec la même amplitude (+0,017, +0,017, +0,017, +0,021 en local contre +0,013, +0,015, +0,016, +0,023 en public).

## 2. Bruit du découpage public / caché (mesuré)

Script : `tasks/Hackathon_Watermark_Localisation_Task/solution/audit_generalisation.py`, sur 235 images officielles tenues hors de l'entraînement des réseaux (prédictions de `probs/*_train.npz`).

| Outil | Images | mIoU moyen | Écart-type par image |
|---|---|---|---|
| PixelSeal | 50 | 0,695 | 0,153 |
| StegaStamp | 95 | 0,847 | 0,146 |
| TrustMark | 90 | 0,647 | 0,161 |

Ces réseaux sont ceux de la nuit (version v1), pas ceux du fichier final ; seul l'écart-type par image (0,153) sert ici.

- Erreur standard d'un score sur 306 images (public) : 0,009. Sur 714 images (caché) : 0,006. Sur les 510 images de validation : 0,007.
- Écart public − caché dû au seul tirage : écart-type 0,0105, soit **± 0,021 à 95 %**.
- Conséquence pour le choix du fichier final : entre deux fichiers, une différence de score public inférieure à environ 0,01 ne veut rien dire. Il faut trancher sur la validation locale (510 images), pas sur le public.

## 3. Les masques prédits sur le test sont-ils réalistes ? (mesuré)

Test sans étiquette : on compare la forme des masques prédits sur les 1020 images de test à celle des 510 vrais masques d'entraînement, par outil. Un post-traitement cassé ou réglé sur le public décale ces distributions.

| Fichier | Couverture moyenne (vérité train : 0,49–0,50) | Images entièrement marquées (vérité : 10–12 %) | Images presque vides (vérité : 12–16 %) | Verdict |
|---|---|---|---|---|
| `submission_baseline.pt` (0,4584 public) | 0,46–0,51 | 0–51 % | 40–49 % | décalé sur les 3 outils : presque tout en « tout ou rien » |
| `submission_t1_v1.pt` (0,7065 public) | 0,48–0,50 | 7–10 % | 14–17 % | conforme sur les 3 outils |
| `submission_ens_amilA_seg2.pt` (non soumis) | 0,50–0,52 | 9–17 % | 12–21 % | StegaStamp conforme ; PixelSeal et TrustMark : trop d'images vides ou pleines |

Lecture : le contrôle sépare bien un fichier faible (baseline) d'un fichier sain (v1). Le fichier d'ensemble non soumis montre le défaut qu'Amil avait mesuré de son côté (nos réseaux JURECA font baisser sa moyenne) : il n'a pas été envoyé, à raison.

## 4. Protocole de validation : pourquoi le score local est fiable

Décrit par la session Modal d'Amil dans `challenge/COORD_T1.md` (messages de 01:05, 01:28, 04:42) ; non re-vérifié dans le code, qui n'est pas dans ce dépôt.

- **Deux séries de réseaux** : la série A apprend sur les images officielles paires et est notée sur les impaires, la série B l'inverse. Chaque image de validation n'est prédite que par la série qui ne l'a pas vue.
- **Réglages (seuils, poids des moyennes, post-traitement SAM) choisis en validation croisée imbriquée** sur les 510 images, jamais sur le score public.
- **Aucune vérité à la décision** : ni masque vrai ni champ `full_watermark` (absent du test) ne servent à produire un masque.
- **Le gain du post-traitement SAM (+0,019 à +0,020)** a été mesuré deux fois indépendamment, sur 5 répétitions.
- Ce qui est autorisé et utilisé au test : l'image, l'outil et le message embarqué (fournis par l'énoncé).

## 5. Risques restants et contrôle à faire sur le fichier final

1. **Fichier final non audité ici.** À lancer sur la machine qui le détient, depuis `tasks/Hackathon_Watermark_Localisation_Task` :
   `python solution/audit_generalisation.py modal_t1/boost2/sam_snap/submission_t1_samsnap_VWY.pt`
   Attendu : « consistent with the train distribution » sur les 3 outils. Un « SHIFT » sur la part d'images vides ou pleines serait le signe d'un post-traitement trop agressif.
2. **Petits effectifs par outil** : 102 images publiques par outil. Un score par outil en public varie de ± 0,03 par hasard ; ne pas en tirer de réglage.
3. **Validation = même distribution que le test ?** Les fonds et la recette de marquage des images officielles train et test sont supposés identiques (même kit). Les écarts local − public de 0,001 à 0,008 le confirment sur 30 % du test ; rien ne permet de le vérifier sur le reste.
4. **Les équipes à égalité avec nous autour de 0,82** sont dans le bruit du découpage (± 0,02) : le rang final sur cette tâche peut bouger de plusieurs places sans que personne n'ait sur-ajusté.

## 6. Reproduire

```
cd tasks/Hackathon_Watermark_Localisation_Task
../../repos/videoseal/.venv/bin/python solution/audit_generalisation.py \
    solution/submission_baseline.pt probs/amil_extra/submission_t1_v1.pt solution/submission_ens_amilA_seg2.pt \
    --probs probs/amil_trustmark_train.npz probs/amil_stegastamp_train.npz probs/seg2_pixelseal_train.npz
```
Durée : quelques secondes, sans GPU. Les fichiers `.pt` et `probs/` sont ignorés par git (lourds) ; ils sont sur le Mac de Matis.
