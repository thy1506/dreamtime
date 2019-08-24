import argparse
import importlib
import os
import subprocess

spec = importlib.util.spec_from_file_location("cli._common",
                                              os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           "../cli/scripts/_common.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

parser = argparse.ArgumentParser(description='cli and gui builder')
parser.add_argument('-d', '--debug', action='store_true',
                    help='Set log level to Debug')
args = parser.parse_args()

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    ## System & Dependencies Check
    c.log.debug("OS : {}".format(c.get_os()))
    c.log.debug("Python version : {}".format(c.get_python_version()))

    if c.get_os() == c.OS.UNKNOWN:
        c.log.fatal("Unknown OS !")
        exit(1)

    if c.get_python_version() < (3, 5):
        c.log.fatal("Unsupported python version !")
        exit(1)

    if not c.check_node():
        c.log.fatal("NodeJs is not install. It's a required dependency !")
        exit(1)

    if not c.check_yarn():
        c.log.fatal("Yarn is not install. It's a required dependency !")
        exit(1)

    with c.cd(".."):
        p1 = subprocess.Popen(["yarn", "dev:gui"])
        p2 = subprocess.Popen(["yarn", "dev:electron"])

        try:
            exit_codes = [p.wait() for p in (p1, p2)]
        except KeyboardInterrupt:
            p1.kill()
            p2.kill()
