#!/bin/bash
# after the PX check fix: DNS queue (running, PID 121434) -> PixArt Part 2 -> pilot (last)
while kill -0 121434 2>/dev/null; do sleep 60; done
echo "F6 finished $(date -u)"
/home/boa-dit-e3c/scripts/run_px.sh > /home/boa-dit-e3c/logs/f5_px_queue.log 2>&1
echo "PX finished $(date -u)"
/home/boa-dit-e3c/scripts/run_pilot.sh > /home/boa-dit-e3c/logs/f5_pilot_queue.log 2>&1
echo "CHAIN_DONE $(date -u)"
