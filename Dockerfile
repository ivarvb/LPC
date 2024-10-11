#operative system
FROM ubuntu:20.04

MAINTAINER Ivar Vargas Belizario "l.ivarvb@gmail.com"
ENV DEBIAN_FRONTEND=noninteractive


######################################
############## Install necessary libs ###############
RUN apt-get update -y && \
    apt-get install -y apt-utils 2>&1 | grep -v "debconf: delaying package configuration, since apt-utils is not installed" && \
    apt-get install -y wget gnupg gnupg2 curl  && \
    apt-get install -y --reinstall systemd  && \
    apt-get install -y python3-pip python3-dev gcc build-essential cmake python3 python3-venv python3-pip python3-scipy libsuitesparse-dev && \
    rm -rf /var/lib/apt/lists/*


# ######################################
# ############## MONGODB ###############
RUN wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | apt-key add -
RUN echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/5.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-5.0.list
RUN apt-get update
RUN apt-get install -y mongodb-org=5.0.2 mongodb-org-database=5.0.2 mongodb-org-server=5.0.2 mongodb-org-shell=5.0.2 mongodb-org-mongos=5.0.2 mongodb-org-tools=5.0.2
# ############## MONGODB ###############
# ######################################



######################################
# ############## Java ############## #
# Install Oracle JDK 17
RUN apt-get install wget -y
RUN wget https://download.oracle.com/java/17/latest/jdk-17_linux-x64_bin.deb
RUN apt-get install ./jdk-17_linux-x64_bin.deb -y
RUN update-alternatives --install "/usr/bin/java" "java" "/usr/lib/jvm/jdk-17/bin/java" 1
RUN java --version
# ############## End Java ############## #
##########################################



###########################################
# ############# Rad-Pleura ############## #
## COPY requirements.txt /app/radpleura/requirements.txt
## WORKDIR /app/radpleura
## RUN pip3 install -r requirements.txt
## COPY . /app/radpleura

## WORKDIR /app/radpleura
## CMD bash RadPleura.sh docker
# ############# Rad-Pleura ############## #
###########################################

## docker build -t radpleura .

## docker run -d -it -v "$(PWD)":/app/radpleura \
##  -v '/yout/path/dir:/app/radpleura/dataraw' \
##  -p 8001:8001 radpleura
