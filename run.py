from subprocess import *
from time import sleep

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


def main():
    TESTCASES = 1000
    cnt = 0
    for i in range(TESTCASES):
        infile = f".\\tools\\in\\{i:04d}.txt"
        proc = run([".\\target\\release\\ahc017.exe","<",infile],shell=True,stdout=open("temp_out.txt","w"),stderr=PIPE,encoding="utf-8")
        get_point = run(f".\\tools\\vis.exe {infile} temp_out.txt",shell=True,stdout=PIPE,stderr=PIPE,encoding="utf-8")
        score = int(get_point.stdout[8:])
        cnt += score
        logger.debug(f"Test {i:04d}: {str(score).ljust(20,' ')} Score sum: {cnt}")

if __name__ == "__main__":
    main()
