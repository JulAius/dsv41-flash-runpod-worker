# DeepSeek-V4.1-Flash-1017 abliterated (GGUF) sur RunPod Serverless

Worker **queue-based** : `llama-server` (llama.cpp PR #30817 avec le support DeepSeek-V4.1) + proxy RunPod
compatible OpenAI (`https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1`).

Image : `ghcr.io/julaius/dsv41-flash-llamacpp:latest`, buildée par `.github/workflows/build.yml`.

## Config recommandée

| | Choix | Pourquoi |
|---|---|---|
| Modèle | [`orcarouter/...-abliterated-GGUF`](https://huggingface.co/orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF), **`Mix-256G`** | KLD 0,051, top-1 93,7 %, PPL +2,4 % par rapport au modèle d'origine. Les versions 1-bit sont très dégradées (Mix-128G : KLD 0,52 ; IQ1_S : PPL ×5) |
| Moteur | llama.cpp, commit `b72050d03` de la PR #30817 | Celui avec lequel orcarouter a testé ses GGUF. vLLM ne lit pas ces GGUF (voir plus bas) |
| GPU | **2× H200 (282 GB)**, pool `HOPPER_141`, 2 GPU / worker | Mix-256G = 235,8 GB en mémoire (sans l'Engram, gardé côté CPU). Config testée par orcarouter : ~70 tok/s en génération, ~110 tok/s en prompt |
| Datacenter | **US-CA-2** (ou EU-FR-1) | 2× H200 dispo en serverless **et** volume *High-performance* |
| Stockage | Network volume High-performance, **370 GB** | 346,5 GB à télécharger. Le volume HP accélère le cold start et la lecture de l'Engram |
| Endpoint | Queue-based | Cold start long + réponses longues : le Load balancing coupe à 5,5 min par requête |

Prix serverless renvoyés par l'API RunPod le 29/10/2026 :

| Config | Mémoire GPU | Prix | Modèle possible |
|---|---|---|---|
| **2× H200** | 282 GB | **13,98 $/h** | Mix-256G (recommandé) |
| 1× B300 | 288 GB | 11,99 $/h | Mix-256G, mais un seul DC (EUR-IS-1, volume standard), non testé par orcarouter |
| 3× RTX PRO 6000 | 288 GB | 10,47 $/h | Mix-256G, réparti sur 3 GPU PCIe : plus lent |
| 1× B200 | 180 GB | 10,79 $/h | Seulement les versions 1-bit (Mix-128G, huihui UD-IQ1_S) |

## Mesures (29/10/2026, 2× H200, US-CA-2, volume HP)

| | Résultat |
|---|---|
| Génération | **44,7 tok/s** (869 tokens en 19,4 s) |
| Lecture du prompt | ~120 tok/s (prompt court de 45 tokens, peu représentatif) |
| Attente, worker chaud | 0,1 s |
| Chargement du modèle, fichiers déjà en cache sur la machine | 102 s |
| Chargement du modèle, machine neuve | Plus de 8 min (voir « Démarrage à froid » plus bas) |
| Téléchargement Mix-256G sur le volume (Pod CPU) | 11 min (~520 MB/s) |

orcarouter annonce ~70 tok/s sur 2× H200. L'écart vient probablement de la lecture de l'Engram sur le volume
réseau : llama.cpp la mappe avec `MADV_RANDOM` (pas de lecture anticipée). D'où `PREWARM_ENGRAM` (mesures 1 et 2 faites sans).

## 1. Network volume + téléchargement

Volume HP de 370 GB dans US-CA-2, puis un Pod CPU monté sur `/runpod-volume` (même chemin qu'en serverless) :

```bash
pip install -U "huggingface_hub[hf_xet]"
hf download orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF \
  --include "Mix-256G/*" \
  --local-dir /runpod-volume/models/dsv41-flash-abliterated
```

Ensuite, supprimer le Pod. `start.py` prend automatiquement
`/runpod-volume/models/dsv41-flash-abliterated/Mix-256G/*-00001-of-*.gguf`.

> Ne pas utiliser le champ **Model** (model caching) de l'endpoint : il télécharge *toutes* les versions du
> repo.

## 2. Endpoint

| Paramètre | Valeur |
|---|---|
| Type | Queue-based |
| Image | `ghcr.io/julaius/dsv41-flash-llamacpp:latest` |
| GPU | Pool `HOPPER_141` (H200), **2 GPU / worker** |
| Active / Max workers | **0** / **1** |
| Idle timeout | 600 s |
| Execution timeout | 3600 s |
| FlashBoot | Activé |
| Network volume | Celui du modèle |
| Container disk | 20 GB |
| CUDA | ≥ 12.8 |

## 3. Variables d'environnement

Toutes optionnelles :

| Variable | Défaut | Rôle |
|---|---|---|
| `VARIANT` | `Mix-256G` | Sous-dossier du modèle sous `MODEL_DIR` |
| `MODEL_DIR` | `/runpod-volume/models/dsv41-flash-abliterated` | |
| `MODEL_PATH` | auto | Chemin explicite du shard `-00001-of-*.gguf` |
| `CTX_SIZE` | `32768` | Contexte total (valeur testée par orcarouter ; le modèle va jusqu'à 1M). À monter si la VRAM le permet |
| `PARALLEL` | `1` | Slots llama-server (le contexte est partagé entre slots) |
| `LAZY_MODE` | `on` | `off` = Engram (110,6 GB) gardé en RAM, si le worker a assez de RAM |
| `PREWARM_ENGRAM` | `on` | Une fois le modèle chargé, lit une fois le fichier Engram (le plus gros shard) pour remplir le cache de pages, s'il y a assez de RAM. Sinon chaque recherche de ligne pendant la génération va jusqu'au volume réseau |
| `LLAMA_ARG_*` | | Toute option llama-server, par ex. `LLAMA_ARG_REASONING_EFFORT=low` |
| `LLAMA_EXTRA_ARGS` | | Flags llama-server supplémentaires |
| `STARTUP_TIMEOUT` | `3600` | Délai max de chargement du modèle (s) |

## 4. Appels

```bash
curl -s https://api.runpod.ai/v2/$ENDPOINT_ID/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"messages":[{"role":"system","content":"Tu es un assistant."},{"role":"user","content":"Bonjour !"}],"max_tokens":2048}}'
# puis GET https://api.runpod.ai/v2/$ENDPOINT_ID/status/<job_id>
```

```python
from openai import OpenAI

client = OpenAI(
    base_url=f"https://api.runpod.ai/v2/{ENDPOINT_ID}/openai/v1",
    api_key=RUNPOD_API_KEY,
    timeout=1800,  # le premier appel peut attendre le cold start
)
stream = client.chat.completions.create(
    model="deepseek-v4.1-flash-abliterated",
    messages=[
        {"role": "system", "content": "Tu es un assistant."},  # recommandé par orcarouter
        {"role": "user", "content": "Bonjour !"},
    ],
    max_tokens=8192,
    stream=True,
    # extra_body={"chat_template_kwargs": {"thinking_mode": "chat"}},  # pour couper le raisonnement
)
for chunk in stream:
    if chunk.choices:
        delta = chunk.choices[0].delta
        print(getattr(delta, "reasoning_content", None) or delta.content or "", end="", flush=True)
```

## Pourquoi pas vLLM

- vLLM ne lit pas les GGUF `deepseek41` : il faut les safetensors (FP8, ~510 GB pour le modèle d'origine).
- huihui-ai et orcarouter ne publient leur version abliterated qu'en GGUF (et MLX pour orcarouter). Les
  safetensors abliterated qui existent viennent d'autres auteurs (dealignai, jcartu, accès sur demande).
- La recette vLLM officielle demande 2× B200 (21,58 $/h) ou 4× H200 (27,96 $/h), plus ~190 GiB de RAM pour
  l'Engram, et un cold start de plus de 300 GB. C'est rentable seulement avec beaucoup d'utilisateurs en
  parallèle.

## Points à vérifier au premier déploiement

- **Démarrage à froid** : ~236 GB à lire depuis le volume, facturés. Au premier test, RunPod a arrêté le
  conteneur au bout de 8 min 03 s : un worker qui n'a pas lancé `runpod.serverless.start` après ~8 min est tué.
  `start.py` lance donc la boucle de jobs tout de suite, et le handler fait attendre chaque job jusqu'à ce que
  `llama-server` soit prêt.
- **Engram sur network volume** : avec `LAZY_MODE=on`, les lignes de la table sont lues à la demande sur le
  volume réseau. orcarouter recommande du NVMe local ; si la génération est lente, essayer `LAZY_MODE=off`.
- **Test local** : `python3 /src/start.py --rp_serve_api` lance une API RunPod locale sur le port 8000
  (`--test_input` ne marche pas : le SDK ne consomme pas les handlers générateurs dans ce mode).
- **Bug llama.cpp #30944** (ouvert) : un token peut être corrompu quand un préfixe de prompt en cache est
  réutilisé. Si tu le vois, ajoute `"cache_prompt": false` dans la requête.
