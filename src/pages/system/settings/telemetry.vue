<template>
  <div class="settings-fields">
    <section class="box box-section">
      <box-section-item
        label="Telemetry"
        description="Send anonymous information about the program that will help us to continue improving it. This includes: Errors, GPU information and performance.">
        <select v-model="currentValue.telemetry.enabled" class="input">
          <option :value="false">Disabled</option>
          <option :value="true">Enabled</option>
        </select>
      </box-section-item>
    </section>
  </div>
</template>

<script>
import { VModel } from '~/mixins'

export default {
  mixins: [VModel],

  data: () => ({}),

  created() {
    $nucleus.track('PAGE_SETTINGS_TELEMETRY')
  }
}
</script>

<style lang="scss" scoped>
p {
  @apply mb-5;
}
</style>


