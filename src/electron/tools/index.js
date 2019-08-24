const _ = require('lodash')
const fs = require('fs')
const { spawn } = require('child_process')
const EventBus = require('js-event-bus')
const gpuInfo = require('gpu-info')
const utils = require('electron-utils')

const debug = require('debug').default('app:electron:tools')
const { Image } = require('image-js')
// const { Caman } = require('caman')

const AppError = require('../modules/error')
const paths = require('./paths')
const config = require('../../nuxt.config')

/**
 * deepTools.
 * Offers a communication channel between the GUI and NodeJS to interact with operating system tools
 */
module.exports = {
  /**
   *
   */
  testError() {
    Error('tools error!')
  },

  /**
   * Returns a list with information of the graphic cards installed in the computer.
   */
  async getGpusList() {
    const devices = await gpuInfo()
    return _.filter(devices, { AdapterCompatibility: 'NVIDIA' })
  },

  /**
   * This is probably not a good cropper, but it's all I have now...
   *
   * @param {Photo} photo
   * @param {HTMLCanvasElement} canvas
   */
  async crop(photo, canvas) {
    const image = Image.fromCanvas(canvas)
    const savePath = photo.getCroppedFile().getPath()

    debug(`Saving cropped photo in ${savePath}`)
    await image.save(savePath)

    if (!fs.existsSync(savePath)) {
      console.warn(
        'It seems that the first crop method has failed, trying the legacy method'
      )

      await this.legacyCrop(photo, canvas)
    } else {
      photo.getCroppedFile().reload()
    }
  },

  /**
   *
   * @param {*} photo
   * @param {*} canvas
   */
  async legacyCrop(photo, canvas) {
    const canvasAsDataURL = canvas.toDataURL(
      photo.getSourceFile().getMimetype(),
      1
    )

    await photo.getCroppedFile().writeDataURL(canvasAsDataURL)

    if (!fs.existsSync(photo.getCroppedFile().getPath())) {
      throw new Error(
        'There was a problem trying to save the cropped photo. Please make sure the program has write permissions.'
      )
    }
  },

  /**
   *
   * @param {*} job
   */
  transform(job) {
    // Independent preferences for the photo
    const preferences = job.getPreferences()

    // Cropped photo
    const inputFilePath = job
      .getPhoto()
      .getCroppedFile()
      .getPath()

    // Final photo
    const outputFilePath = job.getFile().getPath()

    // CLI Args
    const cliArgs = ['--input', inputFilePath, '--output', outputFilePath]

    if ($settings.processing.usePython) {
      // Use the Python script instead of the executable
      cliArgs.unshift('main.py')
    }

    {
      // Preferences
      cliArgs.push('--bsize')
      cliArgs.push(preferences.boobs.size)

      cliArgs.push('--asize')
      cliArgs.push(preferences.areola.size)

      cliArgs.push('--nsize')
      cliArgs.push(preferences.nipple.size)

      cliArgs.push('--vsize')
      cliArgs.push(preferences.vagina.size)

      cliArgs.push('--hsize')
      cliArgs.push(preferences.pubicHair.size)
    }

    if ($settings.processing.device === 'CPU') {
      cliArgs.push('--cpu')
    } else {
      for (const id of $settings.processing.gpus) {
        cliArgs.push(`--gpu`)
        cliArgs.push(id)
      }
    }

    debug('The transformation process has begun!', {
      inputFilePath,
      outputFilePath,
      preferences,
      cliArgs,
      job
    })

    let process
    const bus = new EventBus()

    if ($settings.processing.usePython) {
      // Use the Python script instead of the executable
      process = spawn('python', cliArgs, {
        cwd: paths.getCli()
      })
    } else {
      process = spawn(paths.getCli('dreampower'), cliArgs)
    }

    process.on('error', error => {
      console.error(error)
      bus.emit('error', null, error)
    })

    process.stdout.on('data', data => {
      console.info(`stdout: ${data}`)
      bus.emit('stdout', null, data)
    })

    process.stderr.on('data', data => {
      console.warn(`stderr: ${data}`)
      bus.emit('stderr', null, data)
    })

    process.on('close', code => {
      console.log(`CLI process exited with code ${code}`)
      bus.emit('ready', null, code)
    })

    bus.on('kill', () => {
      process.stdin.pause()
      process.kill()
    })

    return bus
  },

  //
  fs: require('./fs'),

  //
  shell: require('./shell'),

  //
  paths,

  /**
   *
   */
  utils
}
