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
import path from 'path'
import Base from './base'
import dream from '../dream'

export default class extends Base {
  can() {
    if (!$nucleus.isEnabled) {
      return false
    }

    if (_.isNil(_.get($nucleus, 'about.dreamtime.name'))) {
      return false
    }

    if (_.isNil(_.get($nucleus, 'about.dreamtime.title'))) {
      return false
    }

    if (_.isNil(_.get($nucleus, 'about.dreamtime.github'))) {
      return false
    }

    return true
  }

  getName() {
    return $nucleus.about.dreamtime.name
  }

  getTitle() {
    return $nucleus.about.dreamtime.title
  }

  getGithubRepository() {
    return $nucleus.about.dreamtime.github
  }

  getCurrentVersion() {
    return dream.version
  }

  async install(filePath) {
    try {
      $tools.shell.openItem(filePath)
    } catch (err) {
      $tools.shell.openItem(path.dirname(filePath))
    }
  }
}
