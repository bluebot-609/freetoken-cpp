#!/usr/bin/env bash
# Regenerates the Doxygen documentation site and drops a redirect page at
# the repo root so opening the repo lands you straight in the docs.
#
# The real site has to stay in docs_site/html/ -- its pages reference each
# other with paths relative to that directory, so physically moving
# index.html out would break every link on it. This script gives you the
# same effect (open the repo, land on the docs) without breaking anything.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v doxygen >/dev/null 2>&1; then
    echo "error: doxygen not found. Install it first:" >&2
    echo "  sudo apt-get install -y doxygen graphviz" >&2
    exit 1
fi

# Doxygen does not clean its own output directory between runs -- stale
# pages from a prior config (e.g. before docs/roadmap.md etc. moved into
# docs/) would otherwise linger alongside the new ones indefinitely.
rm -rf docs_site
doxygen Doxyfile

cat > index.html <<'EOF'
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=docs_site/html/index.html">
  <title>FreeToken-C++ Docs</title>
</head>
<body>
  <p>Redirecting to <a href="docs_site/html/index.html">the documentation site</a>...</p>
</body>
</html>
EOF

echo "Docs built: docs_site/html/index.html"
echo "Redirect page: index.html (open this at the repo root)"
