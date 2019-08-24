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
import axios from 'axios'
import compareVersions from 'compare-versions'
import Deferred from 'deferred'
import swal from 'sweetalert'
import dream from '../dream'

const debug = require('debug').default('app:modules:update')

const API_URL = 'https://api.github.com/repos/'

export default class {
  constructor() {
    // Can we update this project?
    this.enabled = false

    // Is there an update available?
    this.available = false

    // We are not updating
    this._resetUpdating()

    this.downloadBus = undefined

    this.current = {
      tag_name: '0.0.0'
    }

    this.latest = {
      tag_name: '0.0.0'
    }

    this.http = axios.create({
      baseURL: API_URL
    })
  }

  /**
   * Returns if this provider is activated
   */
  can() {
    return false
  }

  /**
   * Returns the code name of the project.
   * It is usually the lowercase name
   */
  getName() {
    return undefined
  }

  /**
   * Returns the name of the project.
   */
  getTitle() {
    return undefined
  }

  /**
   * Returns the current version of the project
   */
  getCurrentVersion() {
    return '0.0.0'
  }

  /**
   * Returns the domain and repository of the project in Github.
   * Example: private-dreamnet/dreamtime
   */
  getGithubRepository() {
    return undefined
  }

  /**
   * Returns the file name of the latest version
   */
  getUpdateFileName() {
    const platform = $tools.utils.platform({
      macos: 'macos',
      windows: 'windows',
      linux: 'ubuntu'
    })

    const extension = $tools.utils.platform({
      macos: '',
      windows: '.exe',
      linux: '.AppImage'
    })

    return `${this.getTitle()}-v${
      this.latest.tag_name
    }-${platform}-x64${extension}`
  }

  /**
   * Returns the URLs where the latest version can be downloaded
   */
  getUpdateDownloadURLs() {
    const urls = [
      // CDN
      `${$nucleus.urls.cdn}/releases/${this.getName()}/v${
        this.latest.tag_name
      }/${this.getUpdateFileName()}`
    ]

    const asset = _.find(this.latest.assets, {
      name: this.getUpdateFileName()
    })

    if (!_.isNil(asset)) {
      urls.push(asset.browser_download_url)
    }

    return urls
  }

  /**
   * Set information about the update in progress
   *
   * @param {string} text
   * @param {number} progress
   */
  _setUpdating(text = '', progress = -1) {
    this.updating.active = true
    this.updating.text = text

    if (progress >= 0) {
      this.updating.progress = progress
    }
  }

  /**
   * Restart the information about the update in progress
   * (no longer updating)
   */
  _resetUpdating() {
    this.updating = {
      active: false,
      text: undefined,

      progress: 0,
      mbWritten: 0,
      mbTotal: 0
    }
  }

  /**
   * Fetch releases information
   */
  async fetch() {
    // Indicates if the releases information is available
    this.enabled = false

    if (!navigator.onLine) {
      // Offline mode
      return
    }

    if (!this.can()) {
      console.warn(`Update provider is deactivated`)
      return
    }

    try {
      await this.fetchFromAPI()

      this.enabled = true

      debug(`${this.getTitle()} - Update provider initialized!`, {
        provider: this,
        name: this.getName(),
        title: this.getTitle(),
        currentVersion: this.getCurrentVersion(),
        fileName: this.getUpdateFileName(),
        downloadURLs: this.getUpdateDownloadURLs()
      })
    } catch (err) {
      $rollbar.warn(err, {
        project: this.getTitle()
      })

      console.warn(`${this.getTitle()}: Error at fetch releases`, err)

      this.enabled = false
    }
  }

  /**
   *
   */
  async fetchFromAPI() {
    const response = await this.http.get(
      `${this.getGithubRepository()}/releases`
    )

    // Stable versions only
    const releases = _.filter(response.data, {
      draft: false,
      prerelease: false
    })

    if (releases.length === 0) {
      return
    }

    const currentVersion = this.getCurrentVersion()

    // eslint-disable-next-line
    this.latest = releases[0]
    this.current = _.find(releases, { tag_name: `v${currentVersion}` })
    this.available = compareVersions(this.latest.tag_name, currentVersion) === 1

    // this.available = true // TODO: DEBUG
  }

  /**
   * Download the latest software version
   */
  async download() {
    if (!navigator.onLine) {
      swal({
        icon: 'error',
        title: 'Update failed',
        text: `Please verify that you are connected to the Internet and that ${dream.name} has permission to connect.`
      })

      return false
    }

    this._setUpdating('Downloading...', 0)

    const downloadURLs = this.getUpdateDownloadURLs()
    let filePath

    for (const url of downloadURLs) {
      // Try to download it from each mirror
      try {
        // eslint-disable-next-line no-await-in-loop
        filePath = await this._downloadFrom(url)

        if (_.isNil(filePath)) {
          // Apparently the download has been canceled
          this._resetUpdating()
          return false
        }

        // Download completed, now install
        this._setUpdating('Installing...', 0)

        // eslint-disable-next-line no-await-in-loop
        await this.install(filePath)

        return true
      } catch (err) {
        $rollbar.warn(err, { url, provider: this })
        console.warn(`Error downloading/installing from the URL: ${url}`, err)

        continue
      }
    }

    swal({
      icon: 'error',
      title: 'Update failed',
      text: `There was a problem trying to download the update, please verify that you are connected to the Internet. If the problem persists try to configure or temporarily disable your Firewall/Antivirus/VPN.`
    })

    this._resetUpdating()

    return false
  }

  /**
   *
   * @param {string} url
   */
  _downloadFrom(url) {
    debug(`Trying to update from the URL: ${url}`)

    const deferred = new Deferred()

    this.downloadBus = $tools.fs.download(url)

    this.downloadBus.on('progress', payload => {
      this.updating.progress = payload.progress
      this.updating.mbTotal = payload.mbTotal
      this.updating.mbWritten = payload.mbWritten
      // debug(`Progress: ${progress}`)
    })

    this.downloadBus.on('end', filePath => {
      debug('Download finished!', {
        filePath
      })

      this.downloadBus = undefined
      deferred.resolve(filePath)
    })

    this.downloadBus.on('error', err => {
      this.downloadBus = undefined
      deferred.reject(err)
    })

    return deferred.promise()
  }

  /**
   * Cancel the update in progress
   */
  cancel() {
    if (_.isNil(this.downloadBus)) {
      return
    }

    this.downloadBus.emit('cancel')
  }

  /**
   * Install the downloaded update
   * @param {string} filePath
   */
  async install(filePath) {
    // Custom
  }
}
