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

import _ from 'lodash'
import path from 'path'
import filesize from 'filesize'

const debug = require('debug').default('app:modules:platform')

/**
 *
 */
export default {
  /**
   *
   */
  async init() {
    this.gpuDevices = await this.getGpuDevices()

    this.requirements = {
      all: false,
      cli: this.hasCli(),
      checkpoints: this.hasCheckpoints(),
      windowsMedia: await this.hasWindowsMedia()
    }

    this.requirements.all =
      this.requirements.cli &&
      this.requirements.checkpoints &&
      this.requirements.windowsMedia

    this.isLimited = this.getIsLimited()

    window.addEventListener('online', () => {
      this.isLimited = this.getIsLimited()
    })

    window.addEventListener('offline', () => {
      this.isLimited = this.getIsLimited()
    })

    debug('Platform initialized!', {
      gpuDevices: this.gpuDevices,
      requirements: this.requirements,
      isLimited: this.isLimited
    })
  },

  getIsLimited() {
    return !navigator.onLine || !$nucleus.isEnabled
  },

  /**
   *
   */
  async getGpuDevices() {
    try {
      const devices = await $tools.getGpusList()
      return devices
    } catch (error) {
      console.warn(error)
      return []
    }
  },

  /**
   * Verify if the CLI directory is valid
   */
  hasCli() {
    const dirPath = $settings.folders.cli

    if (!_.isString(dirPath)) {
      // And in some extraordinary way,
      // the user managed to change the setting to something invalid :Pepega:
      return false
    }

    if (!$tools.fs.exists(dirPath)) {
      return false
    }

    // One of these files must exist
    const binaries = ['main.py', 'dreampower.exe', 'cli.exe']

    for (const bin of binaries) {
      if ($tools.fs.exists(path.join(dirPath, bin))) {
        return true
      }
    }

    return false
  },

  /**
   * Check if the checkpoints directory is valid
   * You better be valid.....
   *
   * User: Where I download the checkpoints?
   * Developer: Are you kidding me? Fucking fuck! What the fucking shit
   */
  hasCheckpoints() {
    const dirPath = $tools.paths.getCheckpoints()

    if (!$tools.fs.exists(dirPath)) {
      // I guess it's the first time execution...
      return false
    }

    // All these files must exist
    const models = ['cm.lib', 'mm.lib', 'mn.lib']

    for (const modelFile of models) {
      const modelPath = path.join(dirPath, modelFile)

      if (!$tools.fs.exists(modelPath)) {
        // dude... wtf
        return false
      }

      const stats = $tools.fs.stats(modelPath)
      const size = filesize(stats.size, { exponent: 2, output: 'object' })

      if (size.value < 690) {
        // almost... you almost had it
        return false
      }
    }

    // Con-fucking-grats!
    return true
  },

  /**
   *
   */
  async hasWindowsMedia() {
    if (!$tools.utils.is.windows) {
      // Not running Windows ¯\_(ツ)_/¯
      return true
    }

    const version = window.navigator.appVersion
      .split('NT')[1]
      .split(';')[0]
      .trim()

    if (_.toInteger(version) < 10) {
      // Not running Windows 10 ¯\_(ツ)_/¯
      return true
    }

    const value = await $tools.shell.hasWindowsMedia()

    return value
  }
}
