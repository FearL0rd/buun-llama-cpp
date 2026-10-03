#!/usr/bin/env bash
# Rebuild llama-server after the graph-update logging patch (round 26 diagnostic).
cd ~/buun-llama-cpp
if cmake --build build -j 48 > ~/.strata-bench/rebuild-r26.log 2>&1; then
    echo REBUILD_OK >> ~/.strata-bench/rebuild-r26.log
else
    echo REBUILD_FAILED >> ~/.strata-bench/rebuild-r26.log
fi
