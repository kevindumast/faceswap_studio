"""Variables de certificats cassées.

Sur certains PC, SSL_CERT_FILE pointe vers un certificat d'entreprise qui n'existe plus (ex. Cisco Umbrella).
httpx (gradio_client, huggingface_hub) plante alors dès sa création. On ignore la variable dans ce cas précis :
les connexions restent vérifiées avec les certificats standard (certifi) — rien n'est désactivé.
"""
from __future__ import annotations

import os

_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")


def drop_broken_cert_vars() -> list[str]:
    """Retire les variables qui pointent vers un fichier absent. Renvoie leurs noms."""
    dropped = []
    for var in _VARS:
        path = os.environ.get(var)
        if path and not os.path.isfile(path):
            os.environ.pop(var)
            dropped.append(var)
    return dropped
