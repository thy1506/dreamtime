const _ = require('lodash')
const path = require('path')
const { api, is } = require('electron-utils')
const regedit = require('regedit')

// const { download } = require('electron-dl')

module.exports = {
  /**
   * Open the given file in the desktop's default manner.
   * https://electronjs.org/docs/api/shell#shellopenitemfullpath
   *
   * @param {string} fullPath
   */
  openItem(fullPath) {
    return api.shell.openItem(fullPath)
  },

  /**
   * Open the given external protocol URL in the desktop's default manner. (For example, mailto: URLs in the user's default mail agent).
   * https://electronjs.org/docs/api/shell#shellopenexternalurl-options
   *
   * @param {string} url
   */
  openExternal(url) {
    return api.shell.openExternal(url)
  },

  /**
   *
   * @param  {...any} args
   */
  showOpenDialog(...args) {
    return api.dialog.showOpenDialog(...args)
  },

  /**
   *
   * @param  {...any} args
   */
  showSaveDialog(...args) {
    return api.dialog.showSaveDialog(...args)
  },

  /**
   *
   */
  showMessageBox(...args) {
    return api.dialog.showMessageBox(...args)
  },

  /**
   *
   */
  hasWindowsMedia() {
    if (is.windows && !is.dev) {
      regedit.setExternalVBSLocation(
        path.join(path.dirname(api.app.getPath('exe')), 'resources', 'vbs')
      )
    }

    return new Promise(resolve => {
      if (!is.windows) {
        resolve(true)
        return
      }

      const regKey =
        'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Setup\\WindowsFeatures'

      regedit.list(regKey, (err, result) => {
        if (!_.isNil(err)) {
          resolve(false)
          return
        }

        resolve(result[regKey].keys.includes('WindowsMediaVersion'))
      })
    })
  }
}
