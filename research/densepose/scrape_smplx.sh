#!/bin/bash
# Scrape the authenticated SMPL-X download page to discover available sfile names
# (so we can fetch the SMPL-X<->SMPL correspondence / model-transfer pack).
ENVF=~/codes/wan/.env
U_=$(grep '^SMPLX_USER=' "$ENVF" | cut -d= -f2-)
P_=$(grep '^SMPLX_PASS=' "$ENVF" | cut -d= -f2-)
urle () { local LANG=C i x; for (( i=0; i<${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && printf '%s' "${x}" || printf '%%%02X' "'${x}"; done; }
U=$(urle "$U_"); P=$(urle "$P_")
PAGE=$(curl -s --post301 -d "username=$U&password=$P" "https://smpl-x.is.tue.mpg.de/download.php")
echo "=== all sfiles on smplx download page ==="
echo "$PAGE" | grep -oE 'sfile=[^"&]+' | sed 's/^sfile=//' | sort -u
echo "=== transfer/correspondence candidates ==="
echo "$PAGE" | grep -oiE 'sfile=[^"&]*' | grep -iE 'transfer|corr|to_smpl|mano|flame|deform' | sort -u
