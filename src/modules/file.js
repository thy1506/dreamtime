import _ from 'lodash'
import path from 'path'

const debug = require('debug').default('app:modules:file')

export default class File {
  constructor(path) {
    this.reload(path)
  }

  /**
   *
   * @param {*} path
   * @param {*} dataURL
   */
  static async fromDataURL(path, dataURL) {
    await $tools.fs.write(path, dataURL)
    return new this(path)
  }

  /**
   *
   * @param {*} path
   */
  static fromPath(path) {
    return new this(path)
  }

  /**
   *
   */
  static async fromURL(url) {
    const filePath = await $tools.fs.downloadAsync(url, {
      directory: $tools.paths.get('temp')
    })

    return new this(filePath)
  }

  /**
   *
   * @param {*} path
   */
  update(path) {
    this.reload(path)
  }

  /**
   *
   * @param {*} path
   */
  reload(path) {
    if (_.isNil(path)) {
      path = this.getPath()
    }

    const info = $tools.fs.getInfo(path)

    this.name = info.name
    this.ext = info.ext
    this.dir = info.dir
    this.mimetype = info.mimetype
    this.size = info.size
    this._exists = info.exists
  }

  /**
   *
   */
  getPath() {
    return path.join(this.dir, `${this.name}${this.ext}`)
  }

  /**
   *
   */
  getName() {
    return this.name
  }

  /**
   *
   */
  getExt() {
    return this.ext
  }

  /**
   *
   */
  getDir() {
    return this.dir
  }

  /**
   *
   */
  getMimetype() {
    return this.mimetype
  }

  /**
   *
   */
  getSize() {
    return this.size
  }

  /**
   *
   */
  exists() {
    return this._exists
  }

  /**
   *
   */
  remove() {
    if (!this.exists()) {
      return
    }

    $tools.fs.unlink(this.getPath())
    this.reload()
  }

  /**
   *
   */
  async readAsDataURL() {
    if (!this.exists()) {
      return undefined
    }

    const data = await $tools.fs.read(this.getPath(), 'base64')
    return `data:${this.getMimetype()};base64,${data}`
  }

  /**
   *
   */
  async writeDataURL(dataURL) {
    await $tools.fs.writeDataURL(this.getPath(), dataURL)
    this.reload()
  }

  /**
   *
   * @param {*} targetPath
   */
  async copy(targetPath) {
    await $tools.fs.copy(this.getPath(), targetPath)
  }
}
