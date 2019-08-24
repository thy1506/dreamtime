const fs = require('fs')
const _ = require('lodash')
const { uuid, api, is } = require('electron-utils')
const debug = require('debug').default('app:electron:modules:settings')

const tools = require('../tools')

/**
 * User settings.
 * This class is responsible for loading, saving and offering easy access to user settings.
 *
 * Examples:
 * settings.processing.device -> 'CPU'
 *
 * settings.telemetry.enabled = false
 * settings.save()
 */
const settings = {
  /**
   * Initialize the settings
   */
  async init() {
    await this._initDefault()

    this._path = tools.paths.getRoot('settings.json')
    this._settings = {}

    await this._ensure()

    this.load()

    await this._upgrade()
  },

  /**
   *
   */
  async _initDefault() {
    let hasGPU = false

    try {
      hasGPU = (await tools.getGpusList()).length > 0
      // eslint-disable-next-line
    } catch (err) {}

    this._default = {
      version: 2,
      welcome: true,
      user: uuid(),

      processing: {
        device: hasGPU ? 'GPU' : 'CPU',
        gpus: [0],
        usePython: process.env.NODE_ENV === 'development'
      },

      preferences: {
        boobs: {
          size: '1',
          randomize: true,
          progressive: true
        },
        areola: {
          size: '1',
          randomize: false,
          progressive: true
        },
        nipple: {
          size: '1',
          randomize: false,
          progressive: true
        },
        vagina: {
          size: '0.75',
          randomize: true,
          progressive: true
        },
        pubicHair: {
          size: '1',
          randomize: true,
          progressive: true
        },

        executions: 1,
        randomizePreferences: false,
        progressivePreferences: false,

        useWaifu: false, // weebs out 😡👉🚪
        useRestoration: true,
        useCustomMask: false
      },

      notifications: {
        run: false,
        allRuns: true,
        update: true
      },

      folders: {
        cropped: tools.paths.get('temp'),
        models: tools.paths.get('userData', 'models'),
        masks: tools.paths.get('userData', 'masks'),
        cli: tools.paths.getRoot('cli')
      },

      telemetry: {
        enabled: true
      }
    }
  },

  /**
   * Make sure the settings file exists
   */
  async _ensure() {
    if (fs.existsSync(this._path)) {
      // Exists
      return
    }

    try {
      fs.writeFileSync(this._path, JSON.stringify(this._default, null, 2))
    } catch (err) {
      if (is.windows) {
        api.dialog.showErrorBox(
          'The program could not be started',
          `An error occurred while trying to save the settings, please make sure the program has the necessary permissions to write to:\n${this._path}`
        )

        api.app.exit()
      } else {
        api.dialog.showErrorBox(
          'The program could not be started',
          `An error occurred while trying to save the settings, please make sure the program has the necessary permissions to write to:\n${
            this._path
          }.\nDue to the way Ubuntu installs the program you need to grant 777 permissions to the following folder:\n${tools.paths.getRoot()}`
        )

        api.app.exit()
      }
    }
  },

  /**
   * Check if it is necessary to update the settings file.
   * - Ugly code, here we go!
   */
  async _upgrade() {
    const version = this._settings.version || 1
    const currentVersion = this._default.version

    if (currentVersion === version) {
      return
    }

    if (version === 1 && currentVersion === 2) {
      const newSettings = _.cloneDeep(this._settings)

      newSettings.version = 2
      newSettings.preferences = this._default.preferences
      newSettings.notifications = this._default.notifications

      const {
        boobsSize,
        areolaSize,
        nippleSize,
        vaginaSize,
        pubicHairSize
      } = this._settings.preferences

      newSettings.preferences.boobs.size = boobsSize
      newSettings.preferences.areola.size = areolaSize
      newSettings.preferences.nipple.size = nippleSize
      newSettings.preferences.vagina.size = vaginaSize
      newSettings.preferences.pubicHair.size = pubicHairSize

      this.set(newSettings)
    }
  },

  /**
   * Returns the value of the settings in the path
   *
   * @param {string} path
   */
  get(path = '') {
    if (path.length === 0) {
      return this._settings
    }

    return _.get(this._settings, path)
  },

  /**
   * Set a new value in the settings
   *
   * @param {any} path
   * @param {any} payload
   */
  set(path, payload) {
    if (_.isPlainObject(path)) {
      this._settings = path
      this.save()
    }

    this._settings = _.set(this._settings, path, payload)
    this.save()
  },

  /**
   * Load the settings file. If it is already loaded then it refreshes.
   */
  async load() {
    this._settings = JSON.parse(fs.readFileSync(this._path))
    debug('User Settings loaded!', this._settings)
  },

  /**
   * Save the settings.
   * This function is called automatically if you set a first level variable.
   */
  async save() {
    const payload = JSON.stringify(this._settings, null, 2)
    fs.writeFileSync(this._path, payload)

    if (window && window.$rollbar) {
      $rollbar.configure({
        payload: {
          settings: this._settings
        }
      })
    }
  }
}

module.exports = new Proxy(settings, {
  get: (obj, prop) => {
    if (prop in obj) {
      return obj[prop]
    }

    if (prop in obj._settings) {
      return obj._settings[prop]
    }

    return undefined
  },

  set: (obj, prop, value) => {
    if (!_.isNil(obj._settings)) {
      if (prop in obj._settings) {
        obj._settings[prop] = value
        obj.save()

        return true
      }
    }

    obj[prop] = value
    return true
  }
})
