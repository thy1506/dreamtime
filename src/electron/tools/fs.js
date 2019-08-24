const fs = require('fs')
const mime = require('mime-types')
const path = require('path')
const EventBus = require('js-event-bus')
const axios = require('axios')
const { api } = require('electron-utils')
const filesize = require('filesize')
const unzipper = require('unzipper')

const debug = require('debug').default('app:electron:tools:fs')

/**
 * Returns the base64 of a dataURL
 * @param {*} dataURL
 */
function getBase64Data(dataURL) {
  let encoded = dataURL.replace(/^data:(.*;base64,)?/, '')

  if (encoded.length % 4 > 0) {
    encoded += '='.repeat(4 - (encoded.length % 4))
  }

  return encoded
}

module.exports = {
  /**
   *
   * @param {string} filePath
   */
  getInfo(filePath) {
    const exists = this.exists(filePath)
    const mimetype = mime.lookup(filePath)
    const { name, ext, dir } = path.parse(filePath)

    let size

    if (exists) {
      const stats = fs.statSync(filePath)
      size = stats.size / 1000000.0
    }

    return {
      exists,
      name,
      ext,
      dir,
      mimetype,
      size
    }
  },

  /**
   *
   * @param {*} path
   */
  async read(path, encoding = 'utf-8') {
    return fs.readFileSync(path, { encoding })
  },

  /**
   *
   * @param {*} path
   */
  readJSON(path, encoding = 'UTF-8') {
    return JSON.parse(fs.readFileSync(path, { encoding }))
  },

  /**
   *
   * @param {*} path
   * @param {*} dataURL
   */
  async writeDataURL(path, dataURL) {
    const data = getBase64Data(dataURL)
    return fs.writeFileSync(path, data, 'base64')
  },

  /**
   *
   * @param {*} path
   * @param {*} targetPath
   */
  async copy(path, targetPath) {
    return fs.copyFileSync(path, targetPath)
  },

  /**
   *
   */
  async unlink(path) {
    return fs.unlinkSync(path)
  },

  /**
   *
   * @param {string} filePath
   */
  exists(filePath) {
    return fs.existsSync(filePath)
  },

  /**
   * @param {string} filePath
   */
  stats(filePath) {
    return fs.statSync(filePath)
  },

  /**
   *
   * @param {string} zipPath
   * @param {string} targetPath
   */
  extract(zipPath, targetPath) {
    const bus = new EventBus()

    const stream = fs
      .createReadStream(zipPath)
      .pipe(unzipper.Extract({ path: targetPath }))

    let extracted = 0

    stream.on('close', () => {
      bus.emit('end')
    })

    stream.on('data', entryStream => {
      extracted += 1
      const progress = extracted / 4 // TODO: Hardcoded for checkpoints

      debug({
        extracted,
        entryStream
      })

      bus.emit('progress', null, progress)
    })

    stream.on('error', err => {
      bus.emit('error', null, err)
    })

    /*
    const zip = new AdmZip(zipPath)

    zip.extractAllToAsync(targetPath, overwrite, null, progress => {
      bus.emit('progress', null, progress)

      if (progress === 1) {
        bus.emit('end', null, progress)
      }
    })
    */

    return bus
  },

  /**
   *
   */
  download(url, options = {}) {
    const bus = new EventBus()

    options = {
      // showSaveAs: false,
      directory: api.app.getPath('downloads'),
      fileName: undefined,
      ...options
    }

    const fileName =
      options.fileName ||
      path
        .basename(url)
        .split('?')[0]
        .split('#')[0]
    let filePath = path.join(options.directory, fileName)

    const deleteFile = () => {
      if (fs.existsSync(filePath)) {
        debug(`Deleting file ${filePath}`)
        fs.unlinkSync(filePath)
      }
    }

    deleteFile()

    axios
      .request({
        url,
        timeout: 5000,
        responseType: 'stream',
        maxContentLength: -1
      })
      .then(response => {
        const contentLength = response.data.headers['content-length'] || -1
        const mbTotal = filesize(contentLength, {
          exponent: 2,
          output: 'object'
        }).value

        const output = fs.createWriteStream(filePath)
        const stream = response.data

        const cancel = err => {
          stream.destroy(err)
          deleteFile()

          if (!_.isNil(err)) {
            console.warn('Download canceled due to an error', err)
            bus.emit('error', null, err)
          }
        }

        debug('Downloading file and placing it in a writeStream', {
          url,
          fileName,
          filePath,
          contentLength,
          mbTotal,
          exists: fs.existsSync(filePath)
        })

        output.on('error', err => {
          cancel(err)
        })

        stream.on('data', chunk => {
          output.write(Buffer.from(chunk))

          if (contentLength > 0) {
            const progress = output.bytesWritten / contentLength
            const mbWritten = filesize(output.bytesWritten, {
              exponent: 2,
              output: 'object'
            }).value

            bus.emit('progress', null, {
              progress,
              mbWritten,
              mbTotal
            })
          } else {
            const mbWritten = filesize(output.bytesWritten, {
              exponent: 2,
              output: 'object'
            }).value

            bus.emit('progress', null, {
              progress: -1,
              mbWritten,
              mbTotal
            })
          }
        })

        stream.on('end', () => {
          output.end()
          stream.destroy()

          if (!fs.existsSync(filePath)) {
            filePath = undefined
          }

          bus.emit('end', null, filePath)
        })

        stream.on('error', err => {
          cancel(err)
        })

        bus.on('cancel', () => {
          debug('Download canceled!')
          cancel()
        })
      })
      .catch(err => {
        bus.emit('error', null, err)
      })

    return bus
  },

  downloadAsync(url, options = {}) {
    return new Promise((resolve, reject) => {
      const bus = this.download(url, options)

      bus.on('end', filePath => {
        resolve(filePath)
      })

      bus.on('error', err => {
        reject(err)
      })
    })
  }
}
