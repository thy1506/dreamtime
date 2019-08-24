import _ from 'lodash'

const debug = require('debug').default('app:modules:app')

export default {
  /**
   *
   */
  init() {
    this.name = process.env.APP_NAME
    this.version = process.env.APP_VERSION
    this.status = process.env.APP_STATUS
    this.settings = $nucleus
  }
}
