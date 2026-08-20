# Deploying to Streamlit Community Cloud

The app runs from this GitHub repository, but its Snowflake credentials do not:
`.streamlit/secrets.toml` and the `.p8` key are gitignored, so a fresh
Community Cloud deployment starts with no credentials at all. That is the
`StreamlitSecretNotFoundError` / missing-key class of error on first boot.

Community Cloud also has no filesystem you can put the key file on, so the
`private_key_file` path that local development uses cannot work there. The key
travels inline instead, as base64 of the PEM.

## Steps

1. **Generate the secrets block** from your local, working setup:

   ```
   python tools/make_cloud_secrets.py
   ```

   It writes `.streamlit/cloud_secrets.toml` (gitignored) — the same account,
   user, role, warehouse, database and schema, with `private_key_file` replaced
   by a single-line `private_key_base64`. Nothing is printed to the terminal.

2. **Paste it** into the deployed app on [share.streamlit.io], under
   *Manage app → Settings → Secrets*, replacing whatever is there. Save. The
   app reboots on its own.

3. **Delete `.streamlit/cloud_secrets.toml`** once it is pasted.

## Things that go wrong

- **`StreamlitAPIException: Missing Snowflake connection configuration`.** This
  means the deployed app found no `[connections.snowflake]` section — the
  secrets box is empty, holds a different heading, or belongs to a different
  app. The app now replaces this with a message naming what it did find, so a
  wrong heading says so directly. A `[snowflake]` heading and credentials
  pasted with no heading at all are both accepted as well.
- **A secrets block copied straight from local.** `private_key_file` points at
  a `.p8` that does not exist on Community Cloud, and the Snowflake connector
  reads that setting in preference to any inline key — so the inline key is
  never reached. `get_connection()` in `utils/data.py` now blanks the file
  settings out whenever `private_key_base64` is set, but the cleanest secrets
  block simply omits `private_key_file`.
- **A wrapped paste.** `private_key_base64` must be one unbroken line inside
  the quotes. A wrapped one either fails TOML parsing or silently loses
  characters; the decode error names this case.
- **A missing passphrase.** The key in `.streamlit/rsa_key.p8` is encrypted, so
  `private_key_file_pwd` has to travel with it.
- **A Snowflake network policy.** Community Cloud connects from rotating IPs.
  If the Snowflake account restricts logins by IP, the connection fails after
  authentication is otherwise correct — that needs a policy change on the
  Snowflake side, not a secrets change.
- **Committing `environment.yml`.** It is untracked on purpose. It describes
  the Streamlit-in-Snowflake environment and pins nothing; Community Cloud
  reads a conda environment file in preference to `requirements.txt`, which
  would undo the version pinning the app depends on.

## Verifying

The app's home page runs `data_source_preflight()`, which probes each source
relation (Impect dimensions and events, canonical identities, Opta fixtures,
assets and parsed events). All six reading `True` means the deployed
credentials reach everything the pages need.

[share.streamlit.io]: https://share.streamlit.io
