#!/bin/sh
set -e
service ssh start
exec gunicorn --bind 0.0.0.0:8000 wsgi:app
