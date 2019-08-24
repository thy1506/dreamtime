import _ from 'lodash'
import tippy from 'tippy.js'
import { updater, nudify } from '~/modules'

export default {
  directives: {
    tooltip: {
      inserted(el, binding) {
        let settings = {}

        if (_.isString(binding.value)) {
          settings.content = binding.value
        } else {
          settings = binding.value
        }

        tippy(el, settings)
      }
    }
  },

  filters: {},

  data: () => ({
    // $_dream: dream,

    $nudify: nudify,
    $updater: updater,
    $settings
  })
}
