#!/bin/bash

pkill raspivid
pkill gst-launch-1.0

export DISPLAY=:0

raspivid -t 0 -n -w 640 -h 480 -fps 30 -b 2000000 -o - -cs 1 | \
gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=10 pt=96 ! \
udpsink host=172.17.140.56 port=7001 & 

gst-launch-1.0 -v \
udpsrc port=7002 caps="application/x-rtp, encoding-name=H264, payload=96" ! \
rtph264depay ! \
h264parse ! \
decodebin ! \
videoconvert ! \
autovideosink sync=false
