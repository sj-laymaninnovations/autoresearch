# Layman Secrets Setup Utility
## Installation and Usage Guide

The `layman_secrets_setup.py` script provides a novice-friendly guided terminal flow
for storing secrets in the OS credential vault without ever writing values to disk.

---

## Installation

### Step 1 — Install Python dependencies
```bash
pip install keyring pyyaml
```

On some Linux systems you may also need the SecretService backend:
```bash
pip install secretstorage jeepney   # Linux only
```

### Step 2 — Download the setup script
Place `layman_secrets_setup.py` in your project root or a shared tools folder.
It does not need to live inside the git repo — it can live anywhere on the machine.

---

## Usage

### Run the script
```bash
python layman_secrets_setup.py
```

### What happens step by step

**Step 1 — Welcome screen**
Shows OS detection and which vault will be used (Credential Manager / Keychain / SecretService).

**Step 2 — Paste mode**
Terminal prompts you to paste your `SLUG=value` pairs. Example:
```
PROJECT_ALPHA_ANTHROPIC_API_KEY=sk-ant-abc123...
BOX_CLIENT_ID=xyz789
BOX_CLIENT_SECRET=secret456
```
Signal end of input:
- Windows/Linux: press Enter, then Ctrl+Z, then Enter
- macOS: press Enter, then Ctrl+D

Values are read into memory only. Nothing is written to disk at this step.

**Step 3 — Confirmation screen**
Script displays slugs only (values never shown). You confirm before any storage occurs.

**Step 4 — Storage**
Each slug/value pair is stored in the OS vault via `keyring`. Script confirms each one.

**Step 5 — Metadata collection** (optional)
For each stored slug, you can add:
- Human-readable name
- Description
- Expiry date (YYYY-MM-DD)

Press Enter to skip any field.

**Step 6 — Output**
Script writes `secrets.yaml` to the path you specify (default: `./secrets.yaml`).
File contains slug references and metadata only — safe to commit to git.

---

## Verifying stored secrets

### Quick check via Python
```python
import keyring
value = keyring.get_password("layman-secrets", "YOUR_SLUG_HERE")
print("Found" if value else "Not found")
```

### Check all slugs in a secrets.yaml
```python
import keyring
import yaml

with open("secrets.yaml") as f:
    data = yaml.safe_load(f)

for entry in data["secrets"]:
    slug = entry["slug"]
    val = keyring.get_password("layman-secrets", slug)
    status = "✓" if val else "✗ MISSING"
    print(f"  {status}  {slug}")
```

---

## Updating a secret

To rotate or update a secret value, just run the setup utility again with the same slug.
`keyring.set_password` overwrites the existing value. Update `last_updated` in `secrets.yaml` manually or re-run the utility.

---

## Deleting a secret

```python
import keyring
keyring.delete_password("layman-secrets", "SLUG_TO_DELETE")
```

---

## Troubleshooting

### "No recommended backend was available"
Linux only. Install the SecretService backend:
```bash
pip install secretstorage jeepney
```
Or use the file-based fallback (less secure, stores encrypted on disk):
```bash
pip install keyrings.alt
```

### "Access denied" on Windows
Run the terminal as your normal user (not Administrator). Credential Manager is per-user.

### "keyring.errors.KeyringLocked" on macOS
Keychain may be locked. Unlock it via Keychain Access app or:
```bash
security unlock-keychain ~/Library/Keychains/login.keychain-db
```

### Script hangs waiting for input
Make sure you're signaling EOF correctly:
- Windows: Ctrl+Z then Enter (on a new line)
- macOS/Linux: Ctrl+D

---

## Security Notes

- The paste buffer exists in memory only for the duration of the script
- No temp files are created
- The final `secrets.yaml` contains zero secret values
- `keyring` uses OS-level encryption — values are protected by your OS login
- The `layman-secrets` service name namespaces all platform credentials in the vault
- Agents use the same `keyring.get_password("layman-secrets", slug)` call at runtime
