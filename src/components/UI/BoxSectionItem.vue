<template>
  <itemtag :version="version" :tag="tag" :href="href" class="box-section-item">
    <slot name="icon">
      <figure v-if="isURL(icon)" class="item-icon">
        <img :src="icon" />
      </figure>

      <span v-else-if="icon" class="item-icon">{{ icon }}</span>
    </slot>

    <div class="item-text">
      <p class="item-label">{{ label }}</p>

      <slot name="description">
        <p v-if="description" class="item-description">{{ description }}</p>
      </slot>
    </div>

    <div class="item-extra">
      <slot />
    </div>
  </itemtag>
</template>

<script>
import _ from 'lodash'

export default {
  components: {
    itemtag: {
      name: 'itemtag',
      props: ['tag', 'href', 'version'],
      computed: {
        isVisible() {
          if (_.isNil(this.version)) {
            return true
          }

          return this.version === $dream.version
        }
      },
      render(createElement) {
        if (!this.isVisible) {
          return null
        }

        return createElement(
          this.tag,
          {
            props: {
              href: this.href
            }
          },
          this.$slots.default
        )
      }
    }
  },

  props: {
    label: {
      type: String,
      required: true
    },

    description: {
      type: String,
      default: ''
    },

    icon: {
      type: String,
      default: undefined
    },

    href: {
      type: String,
      default: undefined
    },

    version: {
      type: String,
      default: undefined
    }
  },

  computed: {
    tag() {
      let tag = 'div'

      if (!_.isNil(this.href)) {
        tag = 'app-external-link'
      }

      return tag
    }
  },

  methods: {
    isURL(str) {
      if (_.isNil(str)) {
        return false
      }

      if (_.startsWith(str, '~')) {
        return true
      }

      const pattern = new RegExp(
        '^(https?:\\/\\/)?' + // protocol
        '((([a-z\\d]([a-z\\d-]*[a-z\\d])*)\\.)+[a-z]{2,}|' + // domain name
        '((\\d{1,3}\\.){3}\\d{1,3}))' + // OR ip (v4) address
        '(\\:\\d+)?(\\/[-a-z\\d%_.~+]*)*' + // port and path
        '(\\?[;&a-z\\d%_.~+=-]*)?' + // query string
          '(\\#[-a-z\\d_]*)?$',
        'i'
      ) // fragment locator
      return !!pattern.test(str)
    }
  }
}
</script>

