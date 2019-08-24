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
import swal from 'sweetalert'
import Base from './base'
import platform from '../platform'

export default class extends Base {
  can() {
    if (!$nucleus.isEnabled) {
      return false
    }

    if (_.isNil(_.get($nucleus, 'about.checkpoints.name'))) {
      return false
    }

    if (_.isNil(_.get($nucleus, 'about.checkpoints.title'))) {
      return false
    }

    if (_.isNil(_.get($nucleus, 'about.checkpoints.github'))) {
      return false
    }

    return true
  }

  getName() {
    return $nucleus.about.checkpoints.name
  }

  getTitle() {
    return $nucleus.about.checkpoints.name
  }

  getCurrentVersion() {
    if (!platform.requirements.checkpoints || !platform.requirements.cli) {
      return '0.0.0'
    }

    const filePath = $tools.paths.getCheckpoints('version')

    if (!$tools.fs.exists(filePath)) {
      return '0.0.1'
    }

    let version = $tools.fs.read(filePath) || '0.0.1'
    version = version.trim()

    return version
  }

  getGithubRepository() {
    return $nucleus.about.checkpoints.github
  }

  getUpdateFileName() {
    return `v${this.latest.tag_name}.zip`
  }

  getUpdateDownloadURLs() {
    const urls = [
      // CDN
      `${
        $nucleus.urls.cdn
      }/releases/${this.getName()}/${this.getUpdateFileName()}`
    ]

    return urls
  }

  /**
   * Download the latest software version
   */
  async download() {
    const filePath = $tools.paths.get('downloads', this.getUpdateFileName())

    if ($tools.fs.exists(filePath)) {
      try {
        // Download completed, now install
        this._setUpdating('Installing...', 0)

        // eslint-disable-next-line no-await-in-loop
        await this.install(filePath)

        return true
      } catch (err) {
        swal({
          icon: 'error',
          title: 'Update failed',
          text: `We tried to install the file located in the Downloads folder but an error has occurred. If this problem persists try to delete the file and download it again.`
        })

        $rollbar.warn(err, { provider: this })
        console.warn(`Error installing the checkpoints`, err)
      }
    }

    return super.download()
  }

  async install(filePath) {
    const bus = $tools.fs.extract(filePath, $tools.paths.getCli())

    bus.on('progress', value => {
      this.updating.progress = value
    })

    bus.on('end', value => {
      this.updating.progress = value
      this._resetUpdating()

      $tools.utils.api.app.relaunch()
      $tools.utils.api.app.exit()
    })

    bus.on('error', err => {
      throw err
    })
  }
}
