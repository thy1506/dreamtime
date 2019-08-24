const bullet = require('bullet-train-nodejs')
const _ = require('lodash')
const debug = require('debug').default('app:electron:modules:bullet')

const settings = require('./settings')

const instance = {
  init() {
    const config = {
      environmentID: process.env.BULLET_TRAIN_KEY,
      onChange: (old, params) => {}
    }

    if (!_.isNil(settings.user)) {
      // Anonymous User.
      // In the future this could help us to experiment new features only in a group of users.
      // bullet.identify(settings.user)
    }

    bullet.init(config)

    bullet.getValue()
    debug('Bullet Train initialized!', config)
  }
}

instance.init()

module.exports = new Proxy(instance, {
  get(obj, prop) {
    if (prop in obj) {
      return obj[prop]
    }

    if (prop in bullet) {
      return bullet[prop]
    }

    return undefined
  }
})
