FROM python:3.8

WORKDIR /python-docker

### install packages and ffmpeg
RUN apt-get update && apt-get install -y ffmpeg
##update pip
RUN pip install --upgrade pip

COPY requirements.txt .

## INSTALL PYTHON PACKAGES
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

## ADD SSH PACKAGES
RUN apt-get update && apt install -y nano && apt install -y --no-install-recommends dialog && apt install -y --no-install-recommends openssh-server
RUN echo "root:Docker!" | chpasswd
RUN chmod u+x ./entrypoint.sh
COPY sshd_config /etc/ssh/

## OPEN SSH PORT AND FLASK PORT
EXPOSE 8000 2222

ENTRYPOINT [ "./entrypoint.sh" ]
