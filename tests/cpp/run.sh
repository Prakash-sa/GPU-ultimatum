#!/usr/bin/env bash
# Build and run every C++ course program with warnings as errors and sanitizers.
#   docker run --rm -v "$PWD":/w -w /w cuda-cpu-test bash tests/cpp/run.sh
set -uo pipefail
out=$(mktemp -d)
status=0
for f in docs/cpp/src/0[1-8]_*.cpp; do
    name=$(basename "$f" .cpp)
    if [[ $name == 07_parallel ]]; then san="-fsanitize=thread -O1"; else san="-fsanitize=address,undefined -O1"; fi
    if g++ -std=c++20 -Wall -Wextra -Werror -pthread $san "$f" -o "$out/$name" 2> "$out/$name.log" &&
       "$out/$name" > "$out/$name.out" 2>&1; then
        printf 'ok    %-34s %s\n' "$f" "$(tail -1 "$out/$name.out")"
    else
        printf 'FAIL  %s\n' "$f"; cat "$out/$name.log" "$out/$name.out" 2>/dev/null | head -30; status=1
    fi
done
exit $status
