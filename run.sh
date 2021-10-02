#!/bin/bash
ispc render.ispc.cc -o render_ispc.o --opt=fast-math -h render_ispc.h && \
clang++ -g3 -O3 -c -o tasksys.o tasksys.cc && \
clang++ -g3 -O3 -c -o render.o render.cc && \
clang++ -g3 -O3 -o render tasksys.o render.o render_ispc.o -lpthread && \
time ./render > /tmp/out.ppm && \
convert /tmp/out.ppm /tmp/out.png
