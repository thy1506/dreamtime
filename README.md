![](assets/dreamtime.png)

# DreamTime

DreamTime is a friendly user interface that allows you to use [DreamPower](https://github.com/private-dreamnet/dreampower) to transform photos of people into free and **private** entertainment, something that in the past you could only dream of...

![](assets/preview.png)

## Community

Join the social networks of DreamNet, the community interested in developing this technology. You can also join just to talk, make friends or get help:

- [Keybase](https://keybase.io/team/dreamnet) (Our primary communication channel, here you will find the developers)
- [Twitter](https://twitter.com/DreamNetCom)
- [Facebook Group](https://web.facebook.com/groups/812542509140670)
- [Discord](https://imgur.com/a/YlNj44B) :)

## Support

Developing DreamNet applications is time consuming! Help us accelerate development and offer better updates!

[![ko-fi](https://www.ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/R6R2ZSG3)

[![patreon](https://c5.patreon.com/external/logo/become_a_patron_button.png)](https://www.patreon.com/deepmanyy)

## Source Code

This technology was going to arrive sooner or later and we want to keep it open and free for everyone. Help us in our goal of developing more applications with this technology.

- [GitHub](https://github.com/private-dreamnet/dreamtime)
- [NotABug](https://notabug.org/DreamNet/DreamTime)
- [GitGud](https://gitgud.io/dreamnet/dreamtime)

If you want to share or modify this Software please do it for the same purpose as we do and always release the source code of your modifications. Read the [LICENSE](LICENSE) for more information.

---

# 💜 Binaries

## Requirements

- 64 bits OS
- Windows 7 SP1/Windows 8/Windows 10 1803+
- Ubuntu 16.04+
- **8 GB** of RAM or more.

> ⚠ **N versions of Windows 10** require installing the [Media Feature Pack](https://www.microsoft.com/en-us/software-download/mediafeaturepack).

> 🤷‍♂️ We cannot offer a build for **macOS** because we don't have a Mac to compile the project (FeelsPoorMan). Sorry! We are trying to provide all the necessary information so that macOS users can compile their own version.

## GPU Processing Requirements

- NVIDIA Graphics card with minimum [3.5 CUDA compute capability](https://developer.nvidia.com/cuda-gpus)
- [Latest NVIDIA drivers](https://www.nvidia.com/Download/index.aspx)

> 👉 If you do not have an NVIDIA or compatible graphics card you can use CPU processing.

## Download

- [Releases](https://github.com/private-dreamnet/dreamtime/releases)

## Mirrors

- Windows: [DreamNet CDN](https://cdn.dreamnet.tech/releases/dreamtime/v1.0.0/DreamTime-windows-x64.exe)

## F.A.Q.

**Q: Why does the program ask for firewall permissions?**

A: We use [Rollbar](https://rollbar.com/) and [Nucleus](https://nucleus.sh/) to obtain telemetry information of the program mainly for automatic error reporting and to obtain real-time application information.

We do not store any personal information or photos that have been processed with the program, everything is kept securely on your computer, if you prefer you can use the program in offline mode or disable telemetry.

---

# 💻 Development > 🚧 Area only for developers!

## Requirements

- [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive)
- [Python 3.6.2+](https://www.python.org/downloads/release/python-368/)
- [NodeJS 10+](https://nodejs.org/en/)
- [Yarn](https://yarnpkg.com/en/docs/install)

## What should I know?

- [HTML, CSS](https://www.codecademy.com/catalog/language/html-css), [JavaScript](https://www.codecademy.com/catalog/language/javascript): For website development (DreamTime is a website 😮)
- [SCSS](https://sass-lang.com/): CSS Preprocessor
- [TailwindCSS](https://tailwindcss.com/): CSS Framework
- [VueJS](https://vuejs.org/), [NuxtJS](https://nuxtjs.org/): The JavaScript frameworks. _(VueJS is the best #changemymind)_
- [Electron](https://electronjs.org/): The software used to "transform" DreamTime from a website to a desktop program.

## Setup

> All the source code is inside the src/ folder, from now on all the instructions will be based on this folder.

### Quick Setup

Inside the folder `scripts/` you will find python setup scripts for supported operating systems (Windows, MacOS,Ubuntu, Linux) that will allow you to prepare the project for its development.

- **setup.py**: This script will install all the necessary requirements for Python and NodeJS. In theory you only need to run this script once per version.
- **build.py**: This script will compile the project in the `dist/` folder (generate the binaries for CLI and GUI), once finished you can compress the `dist/` folder and share it with the world.
- **dev-start.py**: This script will launch 2 processes, one to start the NuxtJS server (which will compile all the CSS, JavaScript and Vue files) and other that will start Electron that will load the NuxtJS server. Any change you make in `gui/` will be reflected in real time.

To see possible options for these scripts run --help :
```
 python3 setup.py --help
 python3 build.py --help
```
> ⚠ If you use [Anaconda](https://www.anaconda.com/) or some other program that encapsulates Python or NodeJS in their own development environment it is very likely that these scripts fail, you will have to read the content and execute the commands manually.

> ⚠ The NuxtJS server incorporates an HTTP server to view the live application in a web browser, **you must not access the URL indicated by the console**, the application only works when it is opened from Electron.

## Pull Requests

We seek to improve the software in every possible way, if you want to make a contribution you are welcome to make a Pull Request, just make sure you do it to the **canary branch**.

## ⚠ Drastic Changes in v1.0.0

Because DreamTime has been the **victim of users who take the source code and sell it without complying with the LICENSE conditions**, we are forced to implement certain very unfriendly changes for an open-source project, we apologize for all the inconvenience this may cause, these changes are **temporary** until DreamTime has been noticed on social networks to prevent people from falling into a possible SCAM.

To avoid an alert with information from DreamTime
when you start your program, please make sure you **don't change the name or type of license** on your fork.