<template>
  <div class="settings-fields">
    <p class="notification">These are the global preferences, they are applied by default and you can change them independently in each photo.</p>

    <settings-preferences v-model="currentValue.preferences" />
  </div>
</template>

<script>
import { VModel } from '~/mixins'

export default {
  mixins: [VModel],

  created() {
    $nucleus.track('PAGE_SETTINGS_PREFERENCES')
  }
}
</script>

<style lang="scss" scoped>
</style>
