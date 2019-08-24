import _ from 'lodash'

const debug = require('debug').default('app:mixins:vmodel')

/**
 * @mixin
 */
export default {
  props: {
    /**
     * @model
     */
    value: {
      default: null
    }
  },

  data: () => ({
    /**
     * Contains the current value, this variable must be changed within the component.
     */
    currentValue: null
  }),

  created() {
    // Initial value
    this.currentValue = _.cloneDeep(this.value)
  },

  methods: {
    onChange() {
      // nothing
    },

    $forceCurrentValueUpdate() {
      debug('$currentValue has changed, updating the v-model', {
        oldValue: this.value,
        newValue: this.currentValue
      })

      this.$emit('input', this.currentValue)
      this.onChange(this.currentValue)
    }
  },

  watch: {
    // The local value has changed, update the v-model
    currentValue: {
      handler(value) {
        if (_.isEqual(this.value, value)) {
          return
        }

        // this.$forceCurrentValueUpdate()

        debug('$currentValue has changed, updating the v-model', {
          oldValue: this.value,
          newValue: value
        })

        this.$emit('input', value)
        this.onChange(value)
      },
      deep: true
    },

    // The v-model value has changed, update the local value
    value(value) {
      if (_.isEqual(this.currentValue, value)) {
        return
      }

      debug('v-model value has changed, updating currentValue', {
        oldValue: this.currentValue,
        newValue: value
      })

      this.currentValue = value
      this.onChange(value)
    }
  }
}
