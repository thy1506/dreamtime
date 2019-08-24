import _ from 'lodash'
import moment from 'moment'
import Deferred from 'deferred'

import File from '../file'
import Timer from '../timer'
import { rand } from '../helpers'
import WebError from '~/modules/web-error'

const debug = require('debug').default('app:modules:models:photo-job')

export default class PhotoJob {
  constructor(id, photo) {
    this.id = id
    this.photo = photo

    // Transformation Process event bus
    this.process = undefined

    // Output file, this is the photo already transformed!
    this.file = File.fromPath(photo.getFolderPath(this.getFileName()))

    // Clean initialization
    this.reset()

    /*
    this.debug(`Job created`, {
      id: this.id,
      photo: this.photo,
      file: this.file
    })
    */
  }

  /**
   * Restart the information for a possible remake
   */
  reset() {
    // CLI messages
    this.cli = {
      lines: [],
      error: ''
    }

    // Transformation Preferences
    this.preferences = _.cloneDeep(this.photo.preferences)

    this.isLoading = false
    this.hasFailed = false
    this.hasFinished = false

    this.timer = new Timer()
    this.file.remove()
  }

  /**
   *
   * @param {*} message
   * @param  {...any} args
   */
  debug(message, ...args) {
    debug(`[${this.id}] ${message} `, ...args)
  }

  /**
   * The Job has begun
   */
  onStart() {
    this.isLoading = true
    this.timer.start()
  }

  /**
   * The Job has finished successfully
   */
  onFinish() {
    this.isLoading = false
    this.hasFinished = true
    this.timer.stop()

    const activeWindow = $tools.utils.activeWindow()

    if (!activeWindow.isFocused() && $settings.notifications.run) {
      const notification = new Notification(`💭 Run #${this.id} has finished`, {
        body: 'Now you can save the dream'
      })

      notification.onclick = () => {
        activeWindow.focus()
      }
    }
  }

  /**
   * The Job has failed
   */
  onFail() {
    this.isLoading = false
    this.hasFailed = true
    this.timer.stop()
  }

  /**
   * ID. Execution Number
   */
  getId() {
    return this.id
  }

  /**
   *
   */
  getPhoto() {
    return this.photo
  }

  /**
   *
   */
  getPreferences() {
    return this.preferences
  }

  /**
   * Transformed File Name
   */
  getFileName() {
    const now = moment().unix()

    // Original name normalized to avoid problems
    const originalName = _.truncate(
      _.deburr(this.photo.getSourceFile().getName()),
      { length: 30, omission: '' }
    )

    return `${originalName}-${this.id}-${now}-dreamtime.png`
  }

  /**
   *
   */
  getFile() {
    return this.file
  }

  /**
   *
   */
  getCliError() {
    if (_.isNil(this.cli.error) || this.cli.error.length === 0) {
      return { type: 'debug', message: '' }
    }

    const message = this.cli.error

    if (message.includes('Found no NVIDIA driver on your system')) {
      return {
        type: 'debug',
        message:
          "Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from [here](http://www.nvidia.com/Download/index.aspx). If you don't have an NVIDIA GPU please change the Device option in **Settings** to **CPU**."
      }
    }

    if (message.includes('The NVIDIA driver on your system is too old')) {
      return {
        type: 'debug',
        message:
          'The NVIDIA driver installed on your system is too old! Please update the drivers [here](http://www.nvidia.com/Download/index.aspx), if the drivers are up to date then your GPU may not be compatible. You can also change the Device option in **Settings** to **CPU**.'
      }
    }

    if (message.includes('no longer supports this GPU')) {
      return {
        type: 'debug',
        message:
          'Your GPU is not powerful enough to run this program. Please change the Device option in **Settings** to **CPU**.'
      }
    }

    if (message.includes('Buy new RAM!')) {
      return {
        type: 'debug',
        message:
          'Apparently you have run out of RAM on your system! Try a photo of smaller size or free all possible memory.'
      }
    }

    if (message.includes('CUDA out of memory')) {
      return {
        type: 'debug',
        message:
          'Apparently you have run out of RAM on your GPU! Try a photo of smaller size or free all possible GPU use.'
      }
    }

    if (message.includes("codec can't decode byte")) {
      return {
        type: 'debug',
        message:
          'The algorithm had a problem decoding some characters. This is usually caused by being installed in a location with special characters (accents, spaces, etc.). Please reinstall the program in another location.'
      }
    }

    if (message.includes('invalid device ordinal')) {
      return {
        type: 'debug',
        message:
          'A valid GPU device was not found in the indicated GPU option, please try another. You can also change the Device option in **Settings** to **CPU**.'
      }
    }

    if (message.includes('image is not 512 x 512')) {
      return {
        type: 'debug',
        message:
          'The photo is not 512x512, please make sure you have uploaded the correct photo or enable and use the cropper.'
      }
    }

    if (message.includes('loading Python')) {
      return {
        type: 'debug',
        message:
          'There was a problem loading a necessary DreamPower file. It is possible that your installation is corrupt, download the program again and reinstall.'
      }
    }

    return {
      type: 'error',
      message: `The process has been interrupted by an unknown error, this may be caused by a corrupt installation, please check the console for more information.`
    }
  }

  /**
   *
   */
  cancel() {
    if (!this.isLoading || _.isNil(this.process)) {
      return
    }

    this.process.emit('kill')
  }

  /**
   *
   */
  beforeStart() {
    this.customizePreferences()
  }

  /**
   *
   */
  customizePreferences() {
    if (this.preferences.randomizePreferences) {
      // Randomize
      if (this.preferences.boobs.randomize) {
        this.preferences.boobs.size = rand(0.3, 2.0)
      }

      if (this.preferences.areola.randomize) {
        this.preferences.areola.size = rand(0.3, 2.0)
      }

      if (this.preferences.nipple.randomize) {
        this.preferences.nipple.size = rand(0.3, 2.0)
      }

      if (this.preferences.vagina.randomize) {
        this.preferences.vagina.size = rand(0.3, 1.5)
      }

      if (this.preferences.pubicHair.randomize) {
        this.preferences.pubicHair.size = rand(0, 2.0)
      }
    } else if (this.preferences.progressivePreferences) {
      // Progressive
      const add = 0.2 * (this.id - 1)

      if (this.preferences.boobs.progressive) {
        this.preferences.boobs.size = Number.parseFloat(
          this.preferences.boobs.size
        )
        this.preferences.boobs.size += add
        this.preferences.boobs.size = Math.min(this.preferences.boobs.size, 2.0)
      }

      if (this.preferences.areola.progressive) {
        this.preferences.areola.size = Number.parseFloat(
          this.preferences.areola.size
        )
        this.preferences.areola.size += add
        this.preferences.areola.size = Math.min(
          this.preferences.areola.size,
          2.0
        )
      }

      if (this.preferences.nipple.progressive) {
        this.preferences.nipple.size = Number.parseFloat(
          this.preferences.nipple.size
        )
        this.preferences.nipple.size += add
        this.preferences.nipple.size = Math.min(
          this.preferences.nipple.size,
          2.0
        )
      }

      if (this.preferences.vagina.progressive) {
        this.preferences.vagina.size = Number.parseFloat(
          this.preferences.vagina.size
        )
        this.preferences.vagina.size += add
        this.preferences.vagina.size = Math.min(
          this.preferences.vagina.size,
          1.5
        )
      }

      if (this.preferences.pubicHair.progressive) {
        this.preferences.pubicHair.size = Number.parseFloat(
          this.preferences.pubicHair.size
        )
        this.preferences.pubicHair.size += add
        this.preferences.pubicHair.size = Math.min(
          this.preferences.pubicHair.size,
          2.0
        )
      }
    }
  }

  /**
   *
   */
  start() {
    const deferred = Deferred()

    const onSpawnError = error => {
      deferred.reject(
        new WebError(
          'Unable to start DreamPower!',
          `There was a problem trying to start DreamPower, in **Settings** please make sure that the option **DreamPower Folder** is valid, if you are not a developer make sure that the option **Use Python** is disabled.`,
          {
            error,
            type: 'debug'
          }
        )
      )
    }

    this.beforeStart()

    try {
      this.process = $tools.transform(this)
    } catch (error) {
      setTimeout(() => {
        onSpawnError(error)
      }, 0)
      return deferred.promise
    }

    this.process.on('error', error => {
      // Error before starting
      onSpawnError(error)
    })

    this.process.on('stdout', output => {
      // Output generated by the CLI
      output = output
        .toString()
        .trim()
        .split('\n')

      output.forEach(text => {
        this.cli.lines.unshift({
          text,
          css: {}
        })
      })
    })

    this.process.on('stderr', output => {
      // CLI error
      this.cli.lines.unshift({
        text: output,
        css: {
          'text-danger': true
        }
      })

      this.cli.error += `${output}\n`
    })

    this.process.on('ready', code => {
      this.process = undefined

      if (code === 0 || _.isNil(code)) {
        // The process has been completed successfully
        // Reload the output file information.
        this.file.reload()

        if (this.file.exists()) {
          $nucleus.track('DREAM_COMPLETED')
          deferred.resolve()
        } else {
          deferred.reject(
            new WebError(
              `Transformation #${this.id} failed`,
              'DreamPower has reported that the photo has been transformed but the file does not exist! This may be due to a problem saving the photo, verify that DreamPower has write permissions and that your Antivirus is not detecting false positives. It is also possible that DreamPower is ending abruptly due to a major problem.',
              {
                type: 'warning',
                extra: {
                  output: this.cli.lines
                }
              }
            )
          )
        }
      } else {
        const err = this.getCliError()

        deferred.reject(
          new WebError(`Transformation #${this.id} failed`, err.message, {
            error: Error(this.cli.error),
            type: err.type,
            extra: {
              output: this.cli.lines
            }
          })
        )
      }
    })

    return deferred.promise
  }
}
