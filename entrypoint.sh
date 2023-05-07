#!/bin/sh
set -e
/usr/sbin/sshd
exec gunicorn --bind 0.0.0.0:8000 wsgi:app
