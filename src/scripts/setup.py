import argparse
import importlib.util
import logging
import os
import subprocess

spec = importlib.util.spec_from_file_location("cli._common",
                                              os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           "../cli/scripts/_common.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

spec = importlib.util.spec_from_file_location("cli.setup", "../cli/scripts/setup.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def check_dependencies():
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


def gui_setup(args):
    c.log.info("Installing Gui Dependencies")
    with c.cd(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")):
        r = subprocess.run(["yarn", "install"], shell=True) \
            if c.get_os() == c.OS.WIN else subprocess.run(["yarn", "install"])

        if r.returncode != 0:
            c.log.fatal("Gui Dependencies installation failed")
            exit(1)
    c.log.info("Gui Dependencies successfully installed")


def run(args):
    ## System & Dependencies Check
    check_dependencies()

    if args.debug:
        c.log.setLevel(logging.DEBUG)

    ## Cli dependencies
    s.run(args)

    ## Gui dependencies
    gui_setup(args)

    c.log.info("Installation completed!")
    c.log.info("Now you can run the build.py script to compile the project and get an easy-to-use binary")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="cli and gui dependencies setup")
    s.add_arg_parser(parser)
    args = parser.parse_args()
    run(args)
