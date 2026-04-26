# Daily activity log

## Manual log (`daily.md`)

End of day, from the repo root:

```bash
./scripts/daily_log.sh
./scripts/daily_log.sh "optional one-line note"
```

This appends a row to `daily.md` with the date, time, current commit, branch, and last commit message, plus your note if you gave one. Commit `daily.md` when you want that history on GitHub; skip days with nothing to say.

To remember to run it, add a **calendar reminder** or a **shell alias** in your `~/.zshrc`:

```bash
alias logday='~/path/to/diffusion_watermarking/scripts/daily_log.sh'
```

## Automated daily commit (`heartbeat.log`)

A scheduled **GitHub Action** in `.github/workflows/daily_heartbeat.yml` appends a UTC line to `heartbeat.log` and pushes a small commit once per day. After you merge that workflow, enable it and:

1. In the repo: **Settings → Secrets and variables → Actions → New repository secret**
2. Name: `HEARTBEAT_GIT_EMAIL`
3. Value: an email that is **verified** on your GitHub account, same as you use for git (`git config user.email` or the `…@users.noreply.github.com` address from <https://github.com/settings/emails>).

Without that secret, the workflow still runs as `github-actions[bot]`, but those commits often **do not** count on **your** contribution graph. With the secret, the author is `github.actor` and your email so GitHub can attribute the commit to you.

**Try it any time:** **Actions → Daily heartbeat → Run workflow**.

**Local alternative:** `./scripts/heartbeat_local.sh` then `git push` (for example from `cron` on your laptop; keep your token or SSH set up for pushes).

**Caveat:** A repo full of no-op commits is easy to read in history; use only if you still want that trade-off. Scheduled workflows can be **paused** on inactive repos; keep the repo in use or re-enable workflows as needed.
