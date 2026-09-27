"""Certificats TLS sur un PC d'entreprise.

- Sur certains PC, SSL_CERT_FILE pointe vers un certificat d'entreprise qui n'existe plus (ex. Cisco Umbrella) :
  httpx (gradio_client, huggingface_hub) plante alors dès sa création. On ignore la variable dans ce cas précis.
- Un proxy d'entreprise peut aussi re-signer certains sites (ex. *.hf.space signé « Cisco Umbrella ») : sa racine
  est dans le magasin de certificats de Windows, mais pas dans certifi, la liste qu'utilise httpx. Avec truststore,
  Python vérifie avec le magasin du système, comme le navigateur.

Dans les deux cas, les connexions restent vérifiées : rien n'est désactivé.
"""
from __future__ import annotations

import os

_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")


def setup_tls() -> None:
    """À appeler avant toute connexion HTTPS (gradio_client, huggingface_hub)."""
    drop_broken_cert_vars()
    try:
        import truststore
    except ImportError:  # certifi seul : suffit hors proxy d'entreprise
        return
    truststore.inject_into_ssl()


def drop_broken_cert_vars() -> list[str]:
    """Retire les variables qui pointent vers un fichier absent. Renvoie leurs noms."""
    dropped = []
    for var in _VARS:
        path = os.environ.get(var)
        if path and not os.path.isfile(path):
            os.environ.pop(var)
            dropped.append(var)
    return dropped
