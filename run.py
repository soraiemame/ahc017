from subprocess import *
from time import sleep
from sys import stderr

run("cargo build --release",shell=True)

for i in range(10):
    print("Hello!!",file=stderr)
    sleep(1)

