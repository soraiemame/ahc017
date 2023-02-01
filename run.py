from subprocess import *
from time import sleep

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

for i in range(10):
    logger.debug("Hello")
    sleep(1)
