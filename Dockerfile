FROM python:3.8-alpine

WORKDIR /python-docker


# ## update packages && add ffmpeg package
RUN apk add git py-pip python3-dev libffi-dev openssl-dev gcc libc-dev make curl g++
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

### install packages and ffmpeg
RUN apk update && apk add ffmpeg
##update pip
RUN pip install --upgrade pip

COPY requirements.txt .

## INSTALL PYTHON PACKAGES
RUN pip install --no-cache-dir -r requirements.txt 
COPY . .

## ADD SSH PACKAGES
RUN apk add nano && apk --no-install-recommends dialog && apk add --no-install-recommends openssh-server
RUN echo "root:Docker!" | chpasswd
RUN chmod u+x ./entrypoint.sh
COPY sshd_config /etc/ssh/

## OPEN SSH PORT AND FLASK PORT
EXPOSE 8000 2222

ENTRYPOINT [ "./entrypoint.sh" ]
