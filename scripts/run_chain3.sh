#!/bin/bash
# after the cache fix: PixArt Part 2 -> pilot (last)
/home/boa-dit-e3c/scripts/run_px.sh > /home/boa-dit-e3c/logs/f5_px_queue.log 2>&1
echo "PX finished $(date -u)"
/home/boa-dit-e3c/scripts/run_pilot.sh > /home/boa-dit-e3c/logs/f5_pilot_queue.log 2>&1
echo "CHAIN_DONE $(date -u)"
