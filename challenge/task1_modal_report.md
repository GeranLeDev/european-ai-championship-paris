# Task 1 — pipeline Modal (session straycat) : 0,7065 public

État au dim. 22/11, 00:15. Code : `tasks/Hackathon_Watermark_Localisation_Task/modal_t1/`. Les gros fichiers (données fabriquées, poids, probabilités, soumission) restent sur le Mac d'Amil et sur les volumes Modal (`eaic-coco`, `eaic-gen`, `eaic-runs`, `eaic-t1`, espace `snakoffice`).

## Résultat
- **Soumission 1907 (sam. 23:19) : 0,7065 public** (avant : 0,4584). Fichier : `modal_t1/submission_t1_v1.pt` (Mac d'Amil).
- **Validation locale = moitié impaire des images officielles du train** (`int(nom, 16) % 2 == 1`), jamais utilisée pour fabriquer des exemples. Estimation locale 0,705, donc calibrée sur le public. On règle tout en local et on ne soumet que si le gain dépasse 1 point.

| Outil | Réseau seul | Réseau + rectangles |
|---|---|---|
| PixelSeal | 0,559 | 0,571 |
| TrustMark | 0,660 | 0,692 |
| StegaStamp | 0,853 | 0,852 |
| **Moyenne** | 0,691 | **0,705** |

## Méthode
1. **Régime** : décodage du message sur l'image entière ; < 0,80 de bits relus = « partielle », sinon « complète » (`cache/global_{train,test}.json`).
2. **Données fabriquées** avec les embedders du kit officiel (`gen_lib.py`, `app_data.py`) :
   - `/gen/<outil>` : 25 000 exemples sur COCO train2017, zones = contours d'objets COCO, rectangles, barres, taches ;
   - `/gen/<outil>_off` : 25 000 exemples fabriqués **à partir des images officielles paires** (même style que le test). Les zones déjà marquées de l'image source restent étiquetées marquées.
   - Le mélange des deux est nettement meilleur que chacun seul (TrustMark 0,60 → 0,66 ; StegaStamp 0,70/0,82 → 0,85).
3. **Un U-Net par outil** (`train_lib.py`, `app_train.py`, H100, 12 000 pas, lot de 24) : entrée 6 canaux = image + micro-détails (image − flou gaussien, × 20), sortie 128 × 128, perte = entropie croisée + mIoU douce. Le canal de micro-détails est indispensable : sans lui, le calcul en bf16 efface la trace.
4. **PixelSeal** : le U-Net n'apprend rien (0,27), 0,47 avec la trace amplifiée au départ. Ce qui marche : une petite tête de segmentation sur **le détecteur officiel de PixelSeal gelé** + l'accord par position avec le message connu (`pixelseal_research/det_head.py`, 0,56 ; 0,574 avec 25 000 exemples).
5. **Recherche de rectangle** (`app_rect.py`, le `rectsearch.py` de l'équipe sur H100, grille `16,1.25,0.2,2.0`) : le cadre remplace la carte du réseau seulement si son score dépasse un seuil réglé par outil (TrustMark 0,60 ; PixelSeal 0,35 ; StegaStamp jamais).
6. **Jamais de grille vide ni pleine** (`fix_grid`). Combinaison et écriture de la soumission : `combine2.py`.

## Relancer
```bash
cd tasks/Hackathon_Watermark_Localisation_Task/modal_t1      # chemins du Mac d'Amil en dur en tête des fichiers app_*.py
modal run app_data.py::prep                                    # COCO sur Modal (une fois)
modal run app_data.py::main --shards 50 --per 500              # exemples COCO
modal run app_data.py::official --methods trustmark,stegastamp --shards 50 --per 500 --sub _off
modal run app_train.py --methods trustmark,stegastamp --steps 12000 --gen-sub "_off," --tag _mix --val-half
modal run app_rect.py --split test --shards 8 --grid 16,1.25,0.2,2.0 --tag _f
python combine2.py --data <release> --kit <kit> --run pixelseal=...,trustmark=...,stegastamp=... --cache cache_dl --glob-dir cache --rect-tag _f --test out.pt
```

## Ce qui n'a pas marché
- Cartes sans apprentissage (filtre adapté avec la trace recalculée, sorties par position du décodeur) : AUC 0,5 à 0,69.
- Canal « détection » de PixelSeal : AUC 0,56 à 0,58, pas de localisation cachée.
- Entraîner deux fois plus longtemps : TrustMark 0,660 → 0,663.
- Traces JPEG, doublons train/test, messages répétés : rien à exploiter.

## En cours (nuit de samedi à dimanche)
- Diagnostic : où on perd les points, recette exacte des barres et des zones en forme d'objet, puis générateur corrigé (`diag/`, `gen_lib_v2.py`).
- Pistes : tête sur le décodeur TrustMark, retouches finales, zones qui suivent les objets (`tracks/`).
- Deuxième série de réseaux fabriquée à partir des images impaires, puis moyenne de plusieurs architectures.
- Modal plafonne à environ 10 GPU en même temps sur cet espace.

## Coordination Task 2 (important)
- **La collecte `ep` tourne depuis sam. 23:16 sur le Mac d'Amil** : `collect.py --prompts prompts_ref_half.jsonl --endpoints ep --allow-scored --max-per-endpoint 50` → `tasks/model_equality/solution/data/ep_answers.jsonl` (local, ignoré par git). 50 réponses par API, fin vers 02:15. **Ne pas lancer d'autre collecte `ep`** (clé partagée, 4 requêtes à la fois, budget définitif).
- Sondage des `ref` (`tasks/model_equality/solution/straycat/`) : deux fois la même question ne donne pas le même texte, et la latence (environ 30 s partout) ne départage rien.
