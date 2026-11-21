# Consignes orales des organisateurs (mémos vocaux du 21/11)
Transcription automatique (Whisper large-v3-turbo, anglais) dans `challenge/transcripts/` :
- `1335_opening_keynote_fernandez.txt` : 57 min, ouverture + keynote de Pierre Fernandez (Meta) ;
- `1438_task1_intro.txt` : 2 min, intro de la Task 1 ;
- `1440_tasks_rules.txt` : 17 min, les 3 tâches, le compute, le scoreboard et la notation.

La qualité audio est moyenne : certains passages sont brouillés. Ce qui est marqué *(incertain)* est à confirmer sur Discord.
Énoncé écrit : `00_tasks.md`.

## ⚠️ Ce qui compte pour gagner (dit à l'oral, absent du PDF)
- **Le classement final ne dépend pas que du leaderboard.**
  - Le scoreboard public sert de validation. À la fin, il est **gelé**, et seul un membre de l'équipe organisatrice voit les résultats sur le test caché.
  - Les **6 meilleures équipes** sont appelées sur scène, dans un ordre aléatoire, pour présenter, puis sont « grillées » de questions par le jury et le public.
  - **La décision finale combine les scores et la qualité des explications.**
- **Il faut pouvoir expliquer chaque ligne de son code** : « your solution is really yours ». Préparez des slides qui montrent comment chaque tâche a été résolue.
- **Ne pas sur-optimiser le scoreboard.** Les organisateurs citent des équipes qui étaient en tête pendant l'event et ont fini **en bas du classement** sur le test caché.
- **Un seul token de soumission par équipe** : le team lead le reçoit et le partage avec les autres.
- Les questions se posent sur **Discord, dans le canal de la tâche** concernée. Les mentors y répondent, et les **slides y seront uploadées**.
- Environ **100 participants et environ 200 GPU** : à peu près **un GPU A100 par personne**. Un script de setup fourni (« a masterpiece ») évite de tout configurer soi-même.

## Précisions orales par tâche
### Task 1 — Localisation du watermark
- Les mêmes informations que les slides : grille de 128×128 cellules de 4×4 px, label 1 si au moins 50 % de la cellule est marquée, mIoU sur les 2 classes, puis moyenne sur les images.
- **Méthodes : PixelSeal, TrustMark, StegaStamp.**
- Train : 510 images avec masque, méthode et message. Test : 1020 images. L'oral dit « only image » pour le test, alors que la slide dit que la méthode et le message sont aussi fournis : **à vérifier dans les fichiers du dataset**.
- Le keynote juste avant porte sur cette tâche : « **pay very close attention because one of our tasks is based on their work** ».

### Task 2 — Model equality
- Le cadrage : « paie-t-on vraiment le modèle annoncé, ou une version moins chère ? »
- **Modèle différent** : une autre architecture ou un autre modèle de base, une version quantifiée (ex. 4 bits), ou un modèle modifié en aval (LoRA).
- **Même modèle** : un décodage différent, un watermark ajouté aux sorties, ou un system prompt différent.
- Le score « famille » est un petit bonus. **L'essentiel est de bien trancher les paires.**
- Budget : **100 réponses par endpoint**, 1024 tokens d'entrée et 256 en sortie, 4 requêtes en parallèle. Les requêtes invalides ne sont pas décomptées.
- Conseil explicite : avec 1 soumission toutes les 30 min, **construire un setup local d'évaluation** pour itérer plus vite.

### Task 3 — Unlearning
- Le cadre est le droit à l'oubli (RGPD) : les modèles mémorisent les données privées.
- Il y a 2 axes : **forget** (oublier) et **retain** (garder l'utilité). Faire sortir des labels aléatoires « oublie » tout, mais ne sert à rien.
- **Seuil éliminatoire de 70 %.** L'oral parle de « moins de 70 % de ce qu'il avait avant », donc *(incertain)* un seuil peut-être **relatif à l'accuracy d'`orig`**, alors que la slide dit « 70 % d'accuracy held-out ». **Demander sur Discord.** En pratique, rester très près d'`orig` sur le retain.

## Keynote de Pierre Fernandez (Meta) : ce qui sert à la Task 1
Le contenu complet du talk est déjà résumé dans `papers/context/06_organizer_talk_watermarking.md`. Points saillants à l'oral :
- **Pixel Seal :**
  - le watermark est modulé par une **carte JND** : il est **quasi nul dans les zones plates** (ciel) et **fort sur les textures et les bords** ;
  - → pour la localisation, le signal est faible dans les zones uniformes. Il faudra probablement **lisser spatialement ou propager les prédictions** depuis les zones texturées ;
  - entraînement en 3 étapes : force du watermark élevée, puis discriminateur, puis on baisse la force.
- **TrustMark / Pixel Seal** : l'image est **redimensionnée** (256 px) pour l'embedding, puis la distorsion est remise à l'échelle et ajoutée. Le résidu est donc **basse fréquence, à l'échelle 256**.
- **WAM**, la référence directe pour localiser :
  - le watermark est vu comme une **segmentation** : une carte « marqué / pas marqué » par pixel, plus un message par pixel ;
  - à l'entraînement, on tire des **masques aléatoires** pour coller l'image marquée sur l'originale. On obtient à la fois l'augmentation et la vérité terrain du masque ;
  - perte de détection (masque) + perte de décodage ;
  - en multi-messages, **DBSCAN** sur les messages par pixel sépare les zones.
  - → **Idée directe** : générer nos propres données d'entraînement en marquant des images avec PixelSeal ou TrustMark (poids dans `repos/videoseal/ckpts/`) et en les collant par des masques aléatoires. On aurait des milliers de paires (image, masque) en plus des 510 fournies.
  - **StegaStamp n'est pas dans nos poids** : il faut le récupérer.
- **Attaques** :
  - white-box : facile ;
  - black-box : DiffPure (bruit gaussien puis débruitage par diffusion) et compression neuronale / régénération par VAE, qui détruisent les watermarks post-hoc ;
  - un watermark à la génération (autorégressif) résiste mieux.
- **Robustesse ≠ sécurité** : 99,9 % des modifications sont non intentionnelles, donc la robustesse passe d'abord.
- **Questions ouvertes** : rétrocompatibilité entre versions d'extracteur, détecteur partagé entre entreprises, API de détection oui/non attaquable par requêtes.

## Divers
- **Stages au CISPA** : postuler via https://career.cispa.de/internship en indiquant « European AI Championship ». Il y a un pool dédié. **Date limite : 30 novembre.**
- Fraunhofer AISEC et la TUM proposent aussi des thèses, des stages et des postes de PhD.
- C'est la 2e édition. L'an dernier, le circuit passait par Paris, Vienne, Stockholm, Munich, Barcelone et Varsovie, avec une finale.
