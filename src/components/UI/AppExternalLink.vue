<template>
  <a href="#" class="c-app-external-link" @click.prevent="openExternal">
    <slot />
  </a>
</template>

<script>
export default {
  props: {
    href: {
      type: String,
      required: true
    }
  },

  methods: {
    openExternal() {
      $nucleus.track('EXTERNAL_LINK', { href: this.href })
      $tools.shell.openExternal(this.href)
    }
  }
}
</script>

