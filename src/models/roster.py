"""The models of the study in one place: short keys, Hugging Face ids and the named lists of models.

The short key (`gpt`, `qwen36`, ...) is what receipts, hashes and folder names carry, so it never changes.
store.MODELS is the list of keys, in the order that seeds next-state; this module adds each key's Hugging Face id
(from its card profile in generation/model_profiles.py) and the lists a script needs by name. A script takes the
Hugging Face id as input and calls `resolve` to get the key.

    python -m models.roster              the eight study models, one Hugging Face id per line
    python -m models.roster --captured   the seven models whose routers were captured (all but LFM2.5)
    python -m models.roster --judge      the id of the judge that labels sentences and checks answers

To run something for every model:  for m in $(python -m models.roster); do ...; done
"""
import argparse

from generation.model_profiles import CARD_PROFILES
from store import MODELS, require

LFM25_MODEL = 'LiquidAI/LFM2.5-24B-A2B'  # its engine is the steering code's; no card profile, no server script
JUDGE_MODEL = 'Qwen/Qwen3.8-27B'  # not a study model: it labels the sentences and checks the answers

_ids = {**{key: card['model'] for key, card in CARD_PROFILES.items()}, 'lfm25': LFM25_MODEL}
require(set(_ids) == set(MODELS), 'store.MODELS and the card profiles name different models')
HF_IDS = {key: _ids[key] for key in MODELS}  # in store.MODELS order

CAPTURED = tuple(key for key in MODELS if key != 'lfm25')  # the seven models with captured routers
# The order in which the present-state readout walks the captured models; it is part of its frozen specification.
READOUT_ORDER = ('gpt', 'qwen36', 'gemma', 'glm', 'nemotron', 'qwen330b', 'qwen35')
require(sorted(READOUT_ORDER) == sorted(CAPTURED), 'READOUT_ORDER is not the captured models')


def resolve(name: str) -> str:
    """The short key of a model given as its short key or its Hugging Face id."""
    for key, hf_id in HF_IDS.items():
        if name in (key, hf_id):
            return key
    raise ValueError(f'unknown model {name!r}; the study models are {", ".join(HF_IDS.values())}')


def main() -> None:
    parser = argparse.ArgumentParser(prog='models.roster', description=__doc__.split('\n\n')[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true', help='the eight study models (the default)')
    group.add_argument('--captured', action='store_true', help='the seven models whose routers were captured')
    group.add_argument('--judge', action='store_true', help='the judge model')
    args = parser.parse_args()
    if args.judge:
        print(JUDGE_MODEL)
    else:
        print('\n'.join(HF_IDS[key] for key in (CAPTURED if args.captured else MODELS)))


if __name__ == '__main__':
    main()
