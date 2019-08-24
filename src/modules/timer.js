import moment from 'moment'

class Timer {
  constructor() {
    this._start = undefined
    this.duration = 0
    this.interval = undefined
  }

  start() {
    this._start = moment()
    this.duration = 0

    this.interval = setInterval(this.handle.bind(this), 1000)
  }

  handle() {
    this.duration = moment().diff(this._start, 'seconds')

    if (this.duration > 300) {
      // timeout wtf
      this.stop()
    }
  }

  stop() {
    clearInterval(this.interval)
  }
}

export default Timer
