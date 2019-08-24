import argparse
import importlib
import logging
import os
import subprocess
import shutil

spec = importlib.util.spec_from_file_location("cli._common",
                                              os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           "../cli/scripts/_common.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

spec = importlib.util.spec_from_file_location("cli.build", "../cli/scripts/build.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def arg_parser():
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Set log level to Debug')
    args = parser.parse_args()
    return parser, args


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

    if not c.check_pyinstaller():
        c.log.fatal("Pyinstaller is not install. It's a required dependency !")
        exit(1)


def gui_build(args):
    c.log.info('Building Gui')
    with c.cd(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")):
        r = subprocess.run(['yarn', 'build'], shell=True) \
            if c.get_os() == c.OS.WIN else subprocess.run(['yarn', 'build'])
        if r.returncode != 0:
            c.log.fatal("Cli building failed")
            exit(1)
    c.log.info('Gui successfully built')


def run(args):
    ## System & Dependencies Check
    check_dependencies()

    if args.debug:
        c.log.setLevel(logging.DEBUG)

    ## Build Cli
    b.run(args, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../dist"))

    ## Build Gui
    gui_build(args)

    c.log.info('Build completed!')
    c.log.info(
        'It should have generated a folder called dist/, inside you will find the final project files that you can '
        'share with everyone!')
    c.log.info('Enjoy and remember to respect the License!')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='cli and gui builder')
    b.add_arg_parser(parser)
    args = parser.parse_args()
    run(args)
