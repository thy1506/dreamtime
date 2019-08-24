/*
 * DreamTime | (C) 2019 by Ivan Bravo Bravo <ivan@dreamnet.tech>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License 3.0 as published by
 * the Free Software Foundation.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

const { app, BrowserWindow } = require('electron')
const http = require('http')
const path = require('path')
const fs = require('fs')
const contextMenu = require('electron-context-menu')
const utils = require('electron-utils')

const AppError = require('./modules/error')
const { settings, nucleus, rollbar } = require('./modules')
const paths = require('./tools/paths')
const config = require('../nuxt.config')

// Indicate to NuxtJS the root directory of the project
config.rootDir = path.dirname(__dirname)

// Copyright.
// DO NOT DELETE OR ALTER THIS SECTION!
console.log(`
  DreamTime | (C) 2019 by Ivan Bravo Bravo <ivan@dreamnet.tech>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License 3.0 as published by
  the Free Software Foundation.

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <https://www.gnu.org/licenses/>
`)

// Debug
console.log('Starting...')

console.log({
  env: process.env.NODE_ENV,
  paths: {
    getRootPath: utils.getRootPath(),
    appPath: app.getAppPath(),
    exePath: app.getPath('exe'),
    rootPath: paths.getRoot()
  },
  isStatic: utils.pack.isStatic()
})

class DreamApp {
  /**
   * Start the magic!
   */
  static async start() {
    await this.setup()

    this.createWindow()
  }

  /**
   * Prepare the application for use
   */
  static async setup() {
    // https://electronjs.org/docs/tutorial/notifications#windows
    app.setAppUserModelId(process.execPath)

    // https://github.com/sindresorhus/electron-util#enforcemacosapplocation-macos
    utils.enforceMacOSAppLocation()

    // User settings
    await settings.init()

    // Analytics & App settings
    // https://nucleus.sh/docs/gettingstarted
    await nucleus.init()

    // Error reporting
    // https://docs.rollbar.com/docs/nodejs
    rollbar.init()

    //
    this.createModelsDir()

    //
    contextMenu({
      showSaveImageAs: true
    })
  }

  /**
   * Create the program window and load the interface
   */
  static createWindow() {
    // Create the browser window.
    this.window = new BrowserWindow({
      width: 1200,
      height: 700,
      minWidth: 1200,
      minHeight: 700,
      webPreferences: {
        // Script that offers secure communication to the NodeJS API
        preload: path.join(app.getAppPath(), 'electron', 'preload.js')
      }
    })

    // Disable the default menu
    this.window.setMenu(null)

    // Get the interface location
    this.loadURL = this.getNuxtAppLocation()

    if (config.dev) {
      // Development,
      // Load the DevTools and wait for the NuxtJS server to load.
      this.window.webContents.openDevTools()
      this.pollServer()
    } else {
      // Production, load the static interface!
      this.window.loadFile(this.loadURL)
    }
  }

  /**
   * Wait until the NuxtJS server is ready.
   */
  static pollServer() {
    console.log(`Requesting status from the server: ${this.loadURL}`)

    http
      .get(this.loadURL, response => {
        if (response.statusCode === 200) {
          console.log('> Server ready, show time!')
          this.window.loadURL(this.loadURL)
        } else {
          console.log(
            `> The server reported the status code: ${response.statusCode}`
          )
          setTimeout(this.pollServer.bind(this), 300)
        }
      })
      .on('error', () => {
        setTimeout(this.pollServer.bind(this), 300)
      })
  }

  /**
   * Returns the location of the interface
   */
  static getNuxtAppLocation() {
    if (!config.dev) {
      return path.join(config.rootDir, 'dist', 'index.html')
    }

    return `http://localhost:${config.server.port}`
  }

  /**
   * Create the model folder to save the processed photos
   */
  static createModelsDir() {
    const modelsPath = path.join(settings.folders.models, 'Uncategorized')

    if (!fs.existsSync(modelsPath)) {
      fs.mkdirSync(
        modelsPath,
        {
          recursive: true
        },
        error => {
          throw new AppError(
            `Trying to create the directory to save the models,
          please make sure that the application has permissions to create the directory:\n
          ${modelsPath}`,
            error
          )
        }
      )
    }
  }
}

app.on('ready', () => {
  try {
    DreamApp.start()
  } catch (error) {
    rollbar.error(error)
    console.error(error)

    app.quit()
  }
})

app.on('window-all-closed', () => {
  app.quit()
})

app.on('activate', () => {
  DreamApp.createWindow()
})
