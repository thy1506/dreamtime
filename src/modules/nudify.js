import _ from 'lodash'

export default {
  /**
   *
   */
  init() {
    this.reset()
  },

  /**
   *
   */
  reset() {
    this.model = undefined
    this.photo = undefined
  },

  /**
   *
   */
  getModel() {
    return this.model
  },

  /**
   *
   * @param {*} value
   */
  setModel(value) {
    this.model = value
    return this
  },

  /**
   *
   * @param {*} value
   */
  getPhoto() {
    return this.photo
  },

  /**
   *
   * @param {*} value
   */
  setPhoto(value) {
    this.photo = value
  },

  /**
   *
   */
  hasPhoto() {
    return !_.isNil(this.photo)
  },

  /**
   *
   * @param {*} photo
   * @param {*} model
   */
  start(photo, model = null) {
    this.setModel(model)
    this.setPhoto(photo)
  }
}
