#!/bin/bash
#SBATCH --job-name=Portinari  # Job name
#SBATCH --time=20:00:00           # Time limit hrs:min:sec
#SBATCH -w gorgona4
#SBATCH -N 1                        # Number of nodes
#SBATCH --mail-type=ALL
#SBATCH --mail-user=larissa.gomide@dcc.ufmg.br

set -x # all comands are also outputted

# ============================
#  ATIVAÇÃO DO AMBIENTE
# ============================
echo ">> Ativando ambiente: apddv2"
source apddv2/bin/activate

# Variáveis novamente
export HOME="/sonic_home/larissa.gomide/casa/"
export TRANSFORMERS_CACHE="/sonic_home/larissa.gomide/casa/.cache/huggingface"
export CLIP_CACHE="/sonic_home/larissa.gomide/casa/.cache/clip"
export HF_HOME="/sonic_home/larissa.gomide/casa/.cache/huggingface"
export XDG_CACHE_HOME="/sonic_home/larissa.gomide/casa/.cache"
export MPLCONFIGDIR="/sonic_home/larissa.gomide/casa/.matplotlib"

# Gerando scores
echo ">> Rodando: process_dataset_temp.py"
python3 process_dataset_temp.py

deactivate

python3 "/home_cerberus/disk3/larissa.gomide/PKDD/manda_email.py" || echo "Erro ao executar manda_email.py."

hostname   # just show the allocated node

echo "Meu job terminou!"
