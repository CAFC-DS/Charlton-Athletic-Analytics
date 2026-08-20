# =============================================================================
# BUILD THE SECRETS BLOCK FOR STREAMLIT COMMUNITY CLOUD
# =============================================================================
# Community Cloud has no filesystem for the .p8 key, and the key is gitignored
# so it never ships with the repo. This rewrites the local secrets into the
# shape that works there: private_key_file swapped for private_key_base64,
# the same key inlined as a single-line base64 string.
#
#   python tools/make_cloud_secrets.py
#
# The result is written to .streamlit/cloud_secrets.toml (gitignored). Open it,
# copy the whole thing, and paste it into the app's
# Manage app -> Settings -> Secrets box on share.streamlit.io. Nothing is
# printed to the terminal, so the key stays out of your scrollback.
# =============================================================================

import base64
import pathlib
import sys
import tomllib

REPO = pathlib.Path(__file__).resolve().parent.parent
SOURCE = REPO / ".streamlit" / "secrets.toml"
DEST = REPO / ".streamlit" / "cloud_secrets.toml"

# Everything Snowflake needs that is safe to copy across verbatim. The key
# settings are handled separately below; anything else is deliberately dropped
# rather than carried into a deployment that cannot use it.
PASSTHROUGH = ("account", "user", "role", "warehouse", "database", "schema")


def main() -> int:
    if not SOURCE.exists():
        print(f"No {SOURCE} to read. Set up local secrets first.", file=sys.stderr)
        return 1

    snowflake = tomllib.loads(SOURCE.read_text()).get("connections", {}).get("snowflake", {})
    key_file = snowflake.get("private_key_file")
    if not key_file:
        print(
            "[connections.snowflake] has no private_key_file, so there is no key "
            "to inline. This script only converts key-pair auth.",
            file=sys.stderr,
        )
        return 1

    key_path = pathlib.Path(key_file)
    if not key_path.is_absolute():
        key_path = REPO / key_path
    if not key_path.exists():
        print(f"private_key_file points at {key_path}, which does not exist.", file=sys.stderr)
        return 1

    key_base64 = base64.b64encode(key_path.read_bytes()).decode()

    lines = [
        "# Paste this whole block into Manage app -> Settings -> Secrets",
        "# on share.streamlit.io. Do not commit it.",
        "",
        "[connections.snowflake]",
    ]
    lines += [f'{name} = "{snowflake[name]}"' for name in PASSTHROUGH if name in snowflake]
    if snowflake.get("private_key_file_pwd"):
        lines.append(f'private_key_file_pwd = "{snowflake["private_key_file_pwd"]}"')
    lines.append(f'private_key_base64 = "{key_base64}"')

    DEST.write_text("\n".join(lines) + "\n")
    print(f"Wrote {DEST.relative_to(REPO)} ({len(key_base64)} base64 chars of key).")
    print("Copy its contents into the Community Cloud secrets box, then delete it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
