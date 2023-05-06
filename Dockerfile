FROM python:3.8

WORKDIR /python-docker

## CREATE CREDENTIALS file
RUN touch credentials.json
RUN touch token.json

# ## update packages && add ffmpeg package
# RUN apk add py-pip python3-dev libffi-dev openssl-dev gcc libc-dev make curl g++
# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# ENV PATH="/root/.cargo/bin:${PATH}"

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
