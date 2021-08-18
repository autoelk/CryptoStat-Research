#!/bin/bash
export PJ2_HOME=/home/ark/Projects/pj2
export CLASSPATH=$PJ2_HOME/lib
export PATH=`echo $PATH | sed 's/\\/opt\\/jdk[^\\/]*\\/bin/\\/opt\\/jdk1.7\\/bin/'`
export LD_LIBRARY_PATH=/home/ark/Projects/pj2/lib:/usr/lib/x86_64-linux-gnu
export GCCBINDIR=/usr/bin
