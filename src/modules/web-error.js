/*
 * DreamTime | (C) 2019 by Ivan Bravo Bravo <ivan@dreamnet.tech>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License 3.0 as published by
 * the Free Software Foundation.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

import _ from 'lodash'
import { markdown } from 'markdown'
import swal from 'sweetalert'

class WebError extends Error {
  constructor(title, message, opts = {}) {
    super(message)

    opts = {
      error: undefined,
      level: 'error',
      extra: {},
      ...opts
    }

    this.title = title
    this.opts = opts
  }

  report() {
    let { message } = this
    const { title } = this
    const { error, level, extra } = this.opts

    console.log('Reporting error...', {
      message,
      error,
      level,
      weberror: this
    })

    if ($rollbar.isEnabled) {
      const response = $rollbar[level](error || Error(this.message), {
        title,
        message,
        ...extra
      })

      if (response.uuid) {
        message += `
            \nFor more information please report the following URL on [Github](https://github.com/private-dreamnet/dreamtime/issues) or [here](https://git.dreamnet.tech/dreamnet/dreamtime/issues):
            [https://rollbar.com/occurrence/uuid/?uuid=${response.uuid}](https://rollbar.com/occurrence/uuid/?uuid=${response.uuid})`
      } else {
        message += `
            \nFor more information please take a screenshot and report the following on [Github](https://github.com/private-dreamnet/dreamtime/issues) or [here](https://git.dreamnet.tech/dreamnet/dreamtime/issues):\n
            ${error}`
      }
    } else if (error) {
      message += `
            \nFor more information please take a screenshot and report the following on [Github](https://github.com/private-dreamnet/dreamtime/issues) or [here](https://git.dreamnet.tech/dreamnet/dreamtime/issues):\n
            ${error}`
    }

    let icon = 'error'

    if (level === 'warning') {
      icon = 'warning'
    }

    if (level === 'info') {
      icon = 'info'
    }

    const text = document.createElement('div')
    text.innerHTML = markdown.toHTML(message)

    swal({
      title: this.title,
      content: text,
      icon
    })

    const links = document.querySelectorAll('.swal-content a')

    for (const link of links) {
      link.addEventListener('click', e => {
        e.preventDefault()
        $tools.shell.openItem(e.target.href)
      })
    }
  }

  static handle(error) {
    if (!(error instanceof WebError)) {
      error = new WebError(
        'An error has occurred!',
        'Oops! An unknown error has awakened us from our dreams, we will try to solve it in the next version.',
        _.isError(error) ? error : new Error(error)
      )
    }

    error.report()
    return true
  }
}

export default WebError
