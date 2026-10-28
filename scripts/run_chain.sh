#!/bin/bash
# order: E3C Part 1 (running) -> PixArt Part 2 -> DNS / C-DNS -> PixArt pilot (user: "at the end")
while kill -0 93719 2>/dev/null; do sleep 60; done
echo "E3C queue finished $(date -u)"
/home/boa-dit-e3c/scripts/run_px.sh > /home/boa-dit-e3c/logs/f5_px_queue.log 2>&1
echo "PX finished $(date -u)"
/home/boa-dit-dns/scripts/run_f6.sh > /home/boa-dit-dns/logs/f6_queue.log 2>&1
echo "F6 finished $(date -u)"
/home/boa-dit-e3c/scripts/run_pilot.sh > /home/boa-dit-e3c/logs/f5_pilot_queue.log 2>&1
echo "CHAIN_DONE $(date -u)"
