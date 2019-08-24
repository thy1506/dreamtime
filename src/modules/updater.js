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

// eslint-disable-next-line
import _ from 'lodash'

import platform from './platform'
import DreamTime from './update/dreamtime'
import Checkpoints from './update/checkpoints'

const debug = require('debug').default('app:modules:updater')

/**
 * Responsible for obtaining information about the current version,
 * the most recent version and identifying if they are different.
 */
export default {
  /**
   *
   */
  async init() {
    // Software
    this.dreamtime = new DreamTime()
    this.checkpoints = new Checkpoints()

    // Information of the current and most recent version
    await this.dreamtime.fetch()
    await this.checkpoints.fetch()

    if ($settings.notifications.update) {
      if (this.dreamtime.available) {
        const dreamtimeNotification = new Notification(
          `🎉 DreamTime ${this.dreamtime.latest.tag_name} available!`,
          {
            body: 'A new version of DreamTime is available for download.'
          }
        )

        dreamtimeNotification.onclick = () => {
          window.$redirect('/system/about')
          $tools.utils.activeWindow().focus()
        }
      }

      if (platform.requirements.checkpoints && this.checkpoints.available) {
        const checkpointsNotification = new Notification(
          `✨ Checkpoints ${this.checkpoints.latest.tag_name} available!`,
          {
            body: 'A new version of the Checkpoints is available for download.'
          }
        )

        checkpointsNotification.onclick = () => {
          window.$redirect('/system/about')
          $tools.utils.activeWindow().focus()
        }
      }
    }

    debug('Updater initialized!', {
      dreamtime: this.dreamtime,
      checkpoints: this.checkpoints
    })
  }
}
