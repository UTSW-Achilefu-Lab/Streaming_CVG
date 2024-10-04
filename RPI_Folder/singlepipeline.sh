pkill raspivid
pkill gst-launch-1.0

export DISPLAY=:0

# OLD
#gst-launch-1.0 -vvvvv v4l2src device=/dev/video1 ! \
#video/x-raw,width=1920,height=1080,framerate=30/1,format=I420,colorimetry=bt601,interlace-mode=progressive \
#! v4l2h264enc ! video/x-h264,profile=baseline ! h264parse ! rtph264pay config-interval=-1 \
#! udpsink host=172.17.141.30 port=5000

# CURRENTLY TESTING
gst-launch-1.0 -vvvvv v4l2src device=/dev/video1 \
! video/x-raw,width=3280,height=2464,framerate=30/1,format=I420,colorimetry=bt601,interlace-mode=progressive \
! x264enc ! video/x-h264,profile=baseline ! h264parse ! rtph264pay config-interval=-1 \
! udpsink host=172.17.141.30 port=5001 &

#gst-launch-1.0 -v udpsrc address=172.17.141.30 port=5003 caps=application/x-rtp ! \
#rtph264depay ! h264parse ! queue ! v4l2h264dec ! autovideosink sync=false
