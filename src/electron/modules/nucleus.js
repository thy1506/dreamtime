const Nucleus = require('electron-nucleus')
const axios = require('axios')
const debug = require('debug').default('app:electron:modules:nucleus')

const settings = require('./settings')

/**
 * https://nucleus.sh
 * Service that provides us with analytical information and global application settings
 */
const nucleus = {
  isEnabled: false,

  _nucleus: undefined,

  _settings: {},

  /**
   * Initialize the service, this is called automatically
   */
  async init() {
    if (!this.can()) {
      // Can't start the service
      return
    }

    // Nucleus Configuration
    const config = {
      disableTracking: settings.telemetry.enabled === false,
      disableErrorReports: true,
      userId: settings.user,
      version: process.env.APP_VERSION,
      persist: false
    }

    try {
      // Create the Nucleus instance
      this._nucleus = Nucleus(this.getAppId(), config)

      // Get global application settings
      this._settings = (await axios.get(
        `https://nucleus.sh/app/${this.getAppId()}/customdata`
      )).data

      this.isEnabled = true

      debug('Nucleus initialized!', {
        config,
        settings: this._settings
      })
    } catch (err) {
      console.warn('Error at connecting to Nucleus', err)
    }
  },

  /**
   * Returns the APP ID to use the service
   */
  getAppId() {
    return process.env.NUCLEUS_APPID
  },

  /**
   * Returns if the service can be used
   */
  can() {
    return this.getAppId()
  }
}

module.exports = new Proxy(nucleus, {
  get(obj, prop) {
    if (prop in obj) {
      return obj[prop]
    }

    if (prop in obj._settings) {
      return obj._settings[prop]
    }

    if (obj._nucleus && prop in obj._nucleus) {
      if (obj.isEnabled) {
        return obj._nucleus[prop]
      }
    }

    return () => {}
  }
})
