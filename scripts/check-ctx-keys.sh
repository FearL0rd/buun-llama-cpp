#!/usr/bin/env bash
awk '/^\[/{s=$0} /^ctx-size|^yarn|^override-kv/{print s ": " $0}' /home/cesar/models/config.ini | grep -i -E 'coder|flash-next'
