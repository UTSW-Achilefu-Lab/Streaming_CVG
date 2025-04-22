#!/bin/bash

pkill raspivid
pkill gst-launch-1.0

export DISPLAY=:0

# Sender pipeline
# Use Laptop IP address at end

gst-launch-1.0 -vvvv v4l2src device=/dev/video1 ! \
video/x-raw,width=640,height=480,framerate=30/1 ! \
x264enc tune=zerolatency bitrate=512 speed-preset=ultrafast !\
h264parse ! rtph264pay config-interval=-1 ! rtpstreampay ! \
tcpserversink host=172.17.140.252 port=5001 

