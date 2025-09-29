"""Config"""
import logging
import sys
from distutils.util import strtobool
from os import environ
from typing import List

import sentry_sdk

from .logging import InterceptHandler, logger

# basic config, version, cors, debug, key
# NOTE: remember to set the project name, it's useful to debug
# NOTE: remember to set version, it's useful to debug
# NOTE: remember to change secret_key, improve the security
# import os, binascii; binascii.hexlify(os.urandom(12));

SECRET_KEY: str = "25aea18b6f1196bfa74544ab"

ALLOWED_HOSTS: List[str] = ["*"]
DEBUG: bool = bool(strtobool(environ.get("DEBUG", "f")))

PROJECT_NAME: str = "sample-api"
FQDN: str = "sample.ap-mic.com"
TIME_ZONE: str = environ.get("TIME_ZONE", "Asia/Taipei")
STAGE: str = environ.get("STAGE", "dev")

# logging config
LOGGING_LEVEL = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(
    handlers=[InterceptHandler(level=LOGGING_LEVEL)], level=LOGGING_LEVEL
)
logger.configure(handlers=[{"sink": sys.stderr, "level": LOGGING_LEVEL}])

# sentry config
# NOTE: if enable sentry, you need to set dsn
ENABLE_SENTRY: bool = False
SENTRY_DSN = "https://8afd8db1bb8a4e3b9ce474d1eed2da1e@sentry-az.ap-mic.com/7"
if ENABLE_SENTRY and STAGE in ("staging", "prod") and SENTRY_DSN:
    sentry_sdk.init(
        SENTRY_DSN,
        traces_sample_rate=0.2,
        environment=STAGE,
    )
