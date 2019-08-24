class AppError extends Error {
  /**
   *
   * @param {*} message
   */
  constructor(message, error, level = 'error') {
    if (level !== 'debug') {
      if ($rollbar.isEnabled) {
        const response = $rollbar[level](error || new Error(message))

        if (response.uuid) {
          message += `
            \nFor more information please report the following URL on Github or to the developers:\n
            https://rollbar.com/occurrence/uuid/?uuid=${response.uuid}`
        } else {
          message += `
            \nFor more information please take a screenshot and report the following on Github or to the developers:\n
            ${error}`
        }
      } else if (error) {
        message += `
            \nFor more information please take a screenshot and report the following on Github or to the developers:\n
            ${error}`
      }
    }

    super(message)
  }
}

module.exports = AppError
