import _ from 'lodash'
import { uuid } from 'electron-utils/browser'
import swal from 'sweetalert'
import Queue from 'better-queue'
import MemoryStore from 'better-queue-memory'
import File from '../file'
import Timer from '../timer'
import PhotoJob from './photo-job'
import WebError from '../web-error'

const debug = require('debug').default('app:modules:models:photo')

/**
 * Represents the photo to be processed of a Model.
 */
export default class Photo {
  /**
   *
   * @param {Model|null} model Model to which it belongs
   * @param {File} file Instance of the file
   */
  constructor(model, file) {
    // Unique identification and model
    this.uuid = uuid()
    this.model = model

    // Source file, this is the photo that we want to transform
    this.sourceFile = file

    // Cropped file, this is the photo cropped to 512x512
    this.croppedFile = File.fromPath(
      $tools.paths.getCropped(`${this.uuid}.png`)
    )

    // Transformation Preferences
    this.preferences = _.clone($settings.preferences)

    this.reset()

    // Jobs queue
    this.queue = new Queue(
      (job, cb) => {
        job
          .start()
          .then(() => {
            cb(null)
          })
          .catch(err => {
            cb(err)
          })

        return {
          cancel: () => {
            job.cancel()
          }
        }
      },
      {
        maxRetries: 2,
        retryDelay: 1000,
        maxTimeout:
          $settings.processing.device === 'GPU' ? 60 * 1000 : 300 * 1000,
        afterProcessDelay: 1000,
        batchSize: 1,
        store: new MemoryStore()
      }
    )

    this.queue.on('drain', () => {
      debug('All Jobs have finished!')
      this.onFinish()
    })

    this.queue.on('empty', () => {
      // this.onFinish()
    })

    this.queue.on('task_started', (jobId, job) => {
      debug(`The Job #${jobId} has begun!`, { job })
      job.onStart()
    })

    this.queue.on('task_finish', (jobId, job, stats) => {
      if (_.isNil(job)) {
        job = this.getJobById(jobId)
      }

      debug(`The Job #${jobId} has finished!`, { jobId, job, stats })
      job.onFinish()
    })

    this.queue.on('task_failed', (jobId, error, payload) => {
      const job = this.getJobById(jobId)

      console.warn(`The Job #${jobId} has failed!`, { error, payload })
      job.onFail()

      if (_.isError(error)) {
        if (error instanceof WebError) {
          error.report()
        } else {
          throw error
        }
      }
    })

    /*
    this.debug(`Photo created`, {
      uuid: this.uuid,
      model: this.model,
      sourceFile: this.sourceFile,
      croppedFile: this.croppedFile
    })
    */
  }

  reset() {
    //
    this.isLoading = false

    // Transformation Timer
    this.timer = new Timer()

    //
    this.jobs = []
  }

  /**
   *
   * @param {*} message
   * @param  {...any} args
   */
  debug(message, ...args) {
    debug(`${message} `, ...args)
  }

  /**
   * The Photo transformation has begun
   */
  onStart() {
    if (this.isLoading) {
      return
    }

    this.isLoading = true
    this.timer.start()
  }

  /**
   * The Photo transformation has finished
   */
  onFinish() {
    if (!this.isLoading) {
      return
    }

    this.isLoading = false
    this.timer.stop()

    const activeWindow = $tools.utils.activeWindow()

    if (!activeWindow.isFocused() && $settings.notifications.allRuns) {
      const notification = new Notification(`💭 All runs have finished`, {
        body: 'Now you can save all the dreams you like'
      })

      notification.onclick = () => {
        activeWindow.focus()
      }
    }
  }

  /**
   * Returns if the photo is valid
   */
  isValid() {
    return _.isNil(this.getValidationErrorMessage())
  }

  /**
   * Returns the error message for an invalid file. null if there are no errors.
   */
  getValidationErrorMessage() {
    const file = this.getSourceFile()

    if (!file.exists) {
      return 'Apparently the file does not exist anymore!'
    }

    if (
      file.mimetype !== 'image/jpeg' &&
      file.mimetype !== 'image/png' &&
      file.mimetype !== 'image/gif'
    ) {
      return 'The selected file is not a valid photo. Only JPEG, PNG or GIF.'
    }

    return null
  }

  /**
   * Returns the name of the model folder where it will be saved
   */
  getFolderName() {
    return _.isNil(this.model) ? 'Uncategorized' : this.model.name
  }

  /**
   *
   */
  getFolderPath(...args) {
    return $tools.paths.getModels(this.getFolderName(), ...args)
  }

  /**
   * Returns the transformation preferences
   */
  getPreferences() {
    return this.preferences
  }

  /**
   * Return the original file
   */
  getSourceFile() {
    return this.sourceFile
  }

  /**
   * Return the cropped photo
   */
  getCroppedFile() {
    return this.croppedFile
  }

  /**
   *
   * @param {*} id
   */
  getJobById(id) {
    return this.jobs[id - 1]
  }

  /**
   * Start the transformation process!
   */
  async start() {
    if (this.preferences.executions === 0) {
      swal('Invalid Configuration', 'Please set 1 or more runs', 'warning')

      return
    }

    debug(
      `Preparing the transformation for ${this.preferences.executions} jobs`
    )

    this.onStart()

    for (let it = 1; it <= this.preferences.executions; it += 1) {
      const job = new PhotoJob(it, this)

      this.jobs.push(job)
      this.queue.push(job)
    }
  }

  /**
   *
   */
  cancel() {
    if (!this.isLoading) {
      return
    }

    this.jobs.forEach(job => {
      this.queue.cancel(job.id)
    })

    this.onFinish()
  }

  /**
   *
   */
  rerun() {
    this.reset()
    this.start()
  }

  /**
   * Rerun the indicated Job
   */
  rerunJob(id) {
    const job = this.getJobById(id)

    if (_.isNil(job)) {
      return
    }

    debug(`Remaking Job #${id}`)

    job.reset()
    this.onStart()

    this.queue.push(job)
  }
}
