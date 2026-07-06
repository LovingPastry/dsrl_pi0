<!-- ARIS-CODEX:BEGIN -->
## ARIS Codex Skill Scope
ARIS Codex packages installed in this project: skills-codex
Managed entries: 80
Manifest: `.aris/installed-skills-codex.txt`
ARIS repo root: `/home/fuyx/lanzc/Auto-claude-code-research-in-sleep`
Project skill path: `.agents/skills/<skill-name>`
For ARIS Codex workflows, prefer the project-local skills under `.agents/skills/`.
When a skill needs ARIS helper scripts, resolve the repo root from the manifest or set it explicitly:
`ARIS_REPO=$(awk -F'	' '$1=="repo_root"{print $2; exit}' "/home/fuyx/lanzc/dsrl_pi0/.aris/installed-skills-codex.txt")`
Do not edit or delete symlinked skills in place; update upstream or rerun:
`bash /home/fuyx/lanzc/Auto-claude-code-research-in-sleep/tools/install_aris_codex.sh "/home/fuyx/lanzc/dsrl_pi0" --reconcile`
For copied Codex installs, use:
`bash /home/fuyx/lanzc/Auto-claude-code-research-in-sleep/tools/smart_update_codex.sh --project "/home/fuyx/lanzc/dsrl_pi0"`
<!-- ARIS-CODEX:END -->