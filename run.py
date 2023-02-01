from subprocess import *
from time import sleep
from sys import stderr
from logging import getLogger


logger = getLogger()

for i in range(10):
    logger.info("Hello!!")
    sleep(1)

