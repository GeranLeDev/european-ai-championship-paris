# Cluster GPU JURECA (Jülich) — fiche pratique hackathon

Source : `~/Downloads/GPU-usage-paris.pdf` (slides des organisateurs CISPA).

## Matériel
- Environ 200 GPU au total : 40 nœuds de calcul, **1 nœud réservé par job**.
- Nœud de calcul : 4× A100 40 GB, 2× AMD EPYC 7742 (128 cœurs), 512 GB de RAM. Linux x86_64, CUDA 13.
- 12 nœuds de login : 2× Quadro RTX8000 et 1 TB de RAM. Ils servent au code et au debug léger, **pas à l'entraînement**.

## Accès (à faire tôt : la clé SSH met jusqu'à 15 min à s'activer)
1. Créer un compte sur https://judoor.fz-juelich.de/login (bouton Register).
2. Activer le MFA (TOTP : FreeOTP ou Google Authenticator). Il sera demandé à chaque connexion SSH.
3. Signer la *usage agreement* de JURECA (JURECA-DC_GPU) et rejoindre le projet hackathon (`training2666`, « CISPA EU AI Championship 26-27 1 », déjà rattaché au compte geran1 le 21/11).
4. Créer une clé `ssh-keygen -t ed25519` (**id-rsa interdit**). Uploader la clé avec une clause `from="<plages IP>"` : JuDoor propose les plages du réseau d'où l'on se connecte. **Il faut l'uploader depuis le Wi-Fi d'Epitech**, sinon la clé sera refusée sur place. Un VPN ou un partage 4G demande une autre clause.
5. Ajouter dans `~/.ssh/config` :
   ```
   Host juelich
     Hostname jureca.fz-juelich.de
     User geran1
     IdentityFile ~/.ssh/id_ed25519
     MACs hmac-sha2-256-etm@openssh.com
     ControlMaster auto
     ControlPath ~/.ssh/cm-%r@%h:%p
     ControlPersist 12h
   ```
   Les lignes `ControlMaster`/`ControlPersist` (conseil de JuDoor) gardent une connexion persistante : le code MFA n'est demandé qu'une fois, puis les autres terminaux, `scp`/`rsync` et VS Code réutilisent la connexion.
   Se connecter ensuite avec `ssh juelich`, ou via VS Code Remote-SSH.

## Environnement
```bash
jutil env activate -p training2666
cd $PROJECT && mkdir $USER && chmod o-r $USER     # dossier de travail
curl -LsSf https://astral.sh/uv/install.sh | sh
cat << 'EOF' >> ~/.bashrc
export UV_PYTHON_INSTALL_DIR=$PROJECT/$USER/uv/python
export UV_CACHE_DIR=$PROJECT/$USER/uv/cache
export UV_TOOL_DIR=$PROJECT/$USER/uv/tools
export HF_HOME=$PROJECT/$USER/hf                  # ajout perso : poids/datasets hors du $HOME
EOF
exec bash
```
Tout ce qui est volumineux (venvs, cache, checkpoints, datasets) va dans `$PROJECT/$USER`, pas dans `$HOME`, dont le quota est petit.

## Script de soumission `main.sh` (corrigé)
```bash
#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1            # jusqu'à gpu:4 : le nœud entier est réservé de toute façon
#SBATCH --cpus-per-task=128
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --output=output/%j.out
cd $PROJECT/$USER/Hackathon
source .venv/bin/activate
srun python sample_submission.py
```
- `mkdir output` avant le premier `sbatch`, sinon Slurm n'écrit pas les logs.
- Commandes utiles : `sbatch main.sh`, `squeue -u $USER`, `scancel <id>`, `cat output/<id>.out`.

## Format de soumission (leaderboard)
- Le script écrit `submission.csv` au format `id,label`, où `label` est une **probabilité entre 0 et 1**, puis l'envoie au serveur du leaderboard.
- Le sample fourni fait cet envoi depuis le job. La réponse attendue est `{'status': 'success', ...}`.

## Pièges relevés dans les slides
- Projet confirmé sur JuDoor le 21/11 : **training2666** (les slides montraient aussi 2557 et 2614, d’anciennes éditions).
- Slide 21 : `–reservation` est écrit avec un tiret long. Il faut deux tirets `--reservation`, sinon l'option est ignorée.
- Slurm ne cloisonne pas les GPU (bug cgroups) : un job peut voir les 4 GPU. Fixer `CUDA_VISIBLE_DEVICES` ou utiliser explicitement `cuda:0`..`cuda:3`.
- Nœuds de calcul : il n'est pas garanti qu'ils aient un accès internet complet (en général non sur JURECA). Télécharger poids et datasets **depuis le nœud de login** au préalable (checkpoint Pixel Seal, CIFAR-10, modèles HF).

## Portage depuis le Mac (repos/)
| Repo | Sur le Mac | Sur JURECA |
|---|---|---|
| videoseal | MPS, `decord` installé sans dépendances | Le wheel `decord` existe sur Linux x86 → install normale possible ; device `cuda` |
| model-equality-testing | dataset 37 GB non téléchargé | Téléchargeable dans `$PROJECT` (gdown, depuis le login node) |
| SCRUB | MPS, sous-ensemble 20k, ~minutes | A100 : CIFAR-10 complet et SCRUB+R faisables en quelques minutes |

`scrub_smoke.py` détecte déjà CUDA tout seul. Pour Pixel Seal, envoyer explicitement le modèle et les images sur `cuda`. Recréer les venvs sur le cluster (Python 3.10 / 3.11 via `uv`) : les `.venv` du Mac ne sont pas transférables.
