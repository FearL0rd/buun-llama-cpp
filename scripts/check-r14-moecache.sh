#!/usr/bin/env bash
grep -aiE 'moe.cache|moe cache|expert cache|vram.demand|donor' ~/.strata-bench/server-r14-fitfree.log \
  | grep -av 'accepted option\|GET\|POST\|chunk' | head -25
