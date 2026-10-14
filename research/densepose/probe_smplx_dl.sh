#!/bin/bash
# Probe the MPI download host for SMPL-X<->SMPL correspondence / transfer files.
# Tries the authenticated download.php with several candidate sfile names and
# reports HTTP status + content-type + size for each (without saving junk).
ENVF=~/codes/wan/.env
U_=$(grep '^SMPLX_USER=' "$ENVF" | cut -d= -f2-)
P_=$(grep '^SMPLX_PASS=' "$ENVF" | cut -d= -f2-)
urle () { local LANG=C i x; for (( i=0; i<${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && printf '%s' "${x}" || printf '%%%02X' "'${x}"; done; }
U=$(urle "$U_"); P=$(urle "$P_")
DLHOST="https://download.is.tue.mpg.de"
OUT=~/data/bedlam2/dp_assets
cands=(
  "model_transfer.zip"
  "smplx_to_smpl.pkl"
  "smpl_to_smplx.pkl"
  "transfer_data.zip"
  "smplx_mano_flame_correspondences.zip"
  "smplx_kid_template.npy"
)
for dom in smplx sm-plx; do
  for f in "${cands[@]}"; do
    code=$(curl -s -o /tmp/probe.bin -w '%{http_code} %{content_type} %{size_download}' \
      --post-data "username=$U&password=$P" \
      "$DLHOST/download.php?domain=$dom&resume=1&sfile=$f")
    head1=$(head -c 16 /tmp/probe.bin | tr -d '\0' | cat -v | head -c 40)
    echo "[$dom] $f -> $code | head: $head1"
  done
done
