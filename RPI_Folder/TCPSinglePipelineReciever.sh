#!/bin/bash


export DISPLAY=:0

# Recieve pipeline
gst-launch-1.0 -vvvv tcpclientsrc host=172.17.140.56 port=5003 !\
application/x-rtp-stream,encoding-name=H264 ! rtpstreamdepay ! \
rtph264depay ! h264parse ! avdec_h264 ! autovideosink sync=false
