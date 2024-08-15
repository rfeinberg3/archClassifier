# syntax=docker/dockerfile:1

FROM ubuntu:latest 

# Install essentials for buildroot
RUN apt update && \
    apt install -y \ 
    build-essential \
    cmake \ 
    libncurses5-dev \ 
    file \ 
    wget \ 
    cpio \ 
    unzip \ 
    rsync \ 
    bc 
# Install dependencies for target packages
RUN apt install -y \ 
    git \
    subversion 
# Install text editor for debugging
RUN apt install nano

WORKDIR /buildroot

# Enables menuconfig to run on root
ENV FORCE_UNSAFE_CONFIGURE=1

# Copy in buildroot source code
COPY /buildroot /buildroot

# Copy in hexdump script
COPY /hexdump.sh /buildroot/hexdump.sh
RUN chmod +x /buildroot/hexdump.sh






