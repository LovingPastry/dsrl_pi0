#!/bin/bash
# ---------------------------------------------------------------------------
# patch_distrax_jax060.sh
#
# WHY:
#   jax 0.6.0 removed `jax.core.Var` / `jax.core.Literal` (they moved to
#   `jax.extend.core`). distrax 0.1.5 -- the last-ever distrax release (EOL,
#   no newer version exists to upgrade to) -- still references the old
#   location in `_src/utils/transformations.py`. So on jax>=0.6 (which you
#   NEED for Blackwell / RTX 5090 sm_120; the sm_120 codegen + the fp8
#   rounding-mode fix first land in jax/jaxlib 0.6.2), `import distrax` -- and
#   therefore the SAC actor -- crashes with:
#
#     AttributeError: jax.core.Var was removed in JAX v0.6.0.
#                     Use jax.extend.core.Var instead
#
# WHAT THIS DOES:
#   Rewrites those two references in distrax via a version-agnostic try/except
#   shim, so the file works on BOTH jax<0.6 and jax>=0.6. Note a plain string
#   replacement is NOT enough: `import jax` does not make `jax.extend`
#   available, so we must import the symbols explicitly (that is what the shim
#   does). It is idempotent, keeps a `.bak` backup, and verifies the import at
#   the end.
#
# WHEN TO RERUN:
#   After any `pip install` that reinstalls distrax (site-packages edits do not
#   survive a reinstall).
#
# IF A DIFFERENT PACKAGE DOMINOES with a similar error, read the message: it
#   says "... was removed in JAX vX. Use YYY instead." -> map jax.core.XXX to
#   YYY and add the matching import. flax 0.10.2 is expected to be fine (it only
#   uses jax.core.ShapedArray, which still exists on 0.6.2) -- do NOT bump flax,
#   openpi's pi0 depends on it.
#
# USAGE (inside the dsrl_pi0 env, on the machine that has the RTX 5090):
#     bash examples/scripts/patch_distrax_jax060.sh
# ---------------------------------------------------------------------------
set -euo pipefail

python - <<'PY'
import pathlib
import distrax._src.utils.transformations as m

p = pathlib.Path(m.__file__)
s = p.read_text()

# keep a one-time backup so the change is reversible
bak = p.with_suffix(".py.bak")
if not bak.exists():
    bak.write_text(s)
    print(f"[patch] backup written: {bak}")

SHIM = (
    "try:\n"
    "  # jax >= 0.6.0 moved Var/Literal out of jax.core into jax.extend.core\n"
    "  from jax.extend.core import Var as _Var, Literal as _Literal  # pylint: disable=g-import-not-at-top\n"
    "except ImportError:\n"
    "  from jax.core import Var as _Var, Literal as _Literal  # pylint: disable=g-import-not-at-top\n"
)

if "_Var" in s and "_Literal" in s:
    print(f"[patch] already patched: {p}")
else:
    # insert the shim right after distrax's existing linear_util import block,
    # falling back to just after the top-level `import jax` if that changes.
    anchor = "  from jax import linear_util as lu  # pylint: disable=g-import-not-at-top\n"
    if anchor in s:
        s = s.replace(anchor, anchor + "\n" + SHIM, 1)
    else:
        s = s.replace("import jax\n", "import jax\n" + SHIM, 1)
    s = s.replace("jax.core.Var", "_Var").replace("jax.core.Literal", "_Literal")
    p.write_text(s)
    print(f"[patch] patched: {p}")
PY

echo "[patch] verifying 'import distrax' ..."
python -c "import distrax; print('[patch] distrax import OK')"
echo "[patch] done."
