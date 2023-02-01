from subprocess import *
from time import sleep

run("cargo build --release",shell=True)

for i in range(10):
    print("Hello!!")
    sleep(1)

