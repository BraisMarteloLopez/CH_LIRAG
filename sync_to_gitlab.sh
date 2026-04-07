#!/usr/bin/env bash
# =============================================================
# sync_to_gitlab.sh — Mirror completo de GitHub → GitLab
# Uso: ./sync_to_gitlab.sh
#
# Prerequisitos:
#   - git instalado
#   - Acceso SSH o HTTPS a ambos repos
#   - Credenciales GitLab configuradas (ssh-key o token)
# =============================================================
set -euo pipefail

GITHUB_REPO="https://github.com/BraisMarteloLopez/CH_LIRAG.git"
GITLAB_REPO="https://gitlab.com/braismartelolopez/CH_LIRAG.git"
# Para SSH usa: GITLAB_REPO="git@gitlab.com:braismartelolopez/CH_LIRAG.git"

TMPDIR=$(mktemp -d)
echo ">>> Clonando mirror desde GitHub..."
git clone --mirror "$GITHUB_REPO" "$TMPDIR/repo.git"

cd "$TMPDIR/repo.git"

echo ">>> Pushing mirror a GitLab (todas las ramas + tags)..."
git push --mirror "$GITLAB_REPO"

echo ">>> Limpiando directorio temporal..."
rm -rf "$TMPDIR"

echo ">>> Sincronización completada."
echo "    GitHub: $GITHUB_REPO"
echo "    GitLab: $GITLAB_REPO"
