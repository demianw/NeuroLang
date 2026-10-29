#!/bin/sh
# Run one part of the benchmark and write its JSON in $OUT (default /tmp/bench).
#
#   ./run_all.sh micro-native | micro BROWSER | native | queries BROWSER
#                | profile-native | profile BROWSER  (phases of the engine build)
#
# Environment: SITE is the REPL page with its bundle (default /tmp/repl_site),
# PY a Python with NeuroLang's dependencies, NODE_PATH where Playwright is.
# Run the parts one after the other: they compete for the machine.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
OUT=${OUT:-/tmp/bench}
SITE=${SITE:-/tmp/repl_site}
PY=${PY:-/Users/bot/NeuroLang-llmex/.venv/bin/python}
export NODE_PATH=${NODE_PATH:-/Users/bot/node_modules}
export PYTHONPATH=$(ls -d "$ROOT"/packages/* | tr '\n' ':')
export SSL_CERT_FILE=${SSL_CERT_FILE:-/etc/ssl/cert.pem}
mkdir -p "$OUT"
"$PY" "$HERE/queries.py" "$OUT/queries.json" >/dev/null
case "$1" in
  micro-native) "$PY" "$HERE/micro.py" > "$OUT/micro_native.json" ;;
  micro) node "$HERE/browser.js" micro "$2" "$OUT/micro_$2.json" ;;
  native) "$PY" "$HERE/native.py" "$OUT/queries.json" "$OUT/native.json" 2 ;;
  queries)
    repeats=1; [ "$2" = chromium ] && repeats=2
    node "$HERE/browser.js" queries "$2" "$SITE" "$OUT/queries.json" "$OUT/queries_$2.json" $repeats ;;
  profile-native) "$PY" "$HERE/profile_build.py" > "$OUT/profile_native.txt" ;;
  profile) node "$HERE/browser.js" profile "$2" "$SITE" "$OUT/profile_$2.json" ;;
  *) echo "usage: $0 micro-native | micro BROWSER | native | queries BROWSER | profile-native | profile BROWSER"; exit 2 ;;
esac
