"""Narrow environment correction after preparation job 59416713 exited 127.

The inherited `module` function calls `_module_raw` on this host. Preserve that
one supporting function alongside the frozen credential-free allowlist.
"""


def safe_environment(env, additions=None):
    from utility_j1_chain_v1 import safe_environment as original
    result = original(env, additions)
    key = 'BASH_FUNC__module_raw%%'
    if key in env:
        result[key] = env[key]
    return result
