# pylint: skip-file
"""Gunicorn configuration."""
bind = ':{}'.format(8000)

workers = 1
timeout = 90
#worker_class = 'gevent'

accesslog = '-'
