<template>
  <div class="settings-fields">
    <section class="box box-section">
      <box-section-item
        label="On Run"
        description="Show a notification when the program is not in the foreground and a single run have finished.">
        <select v-model="currentValue.notifications.run" class="input">
          <option :value="true">Enabled</option>
          <option :value="false">Disabled</option>
        </select>
      </box-section-item>

      <box-section-item
        label="On Dream"
        description="Show a notification when the program is not in the foreground and all the runs have finished.">
        <select v-model="currentValue.notifications.allRuns" class="input">
          <option :value="true">Enabled</option>
          <option :value="false">Disabled</option>
        </select>
      </box-section-item>

      <box-section-item
        label="On Update"
        :description="`Show a notification when there is a new version of ${$dream.name} or the Checkpoints.`">
        <select v-model="currentValue.notifications.update" class="input">
          <option :value="true">Enabled</option>
          <option :value="false">Disabled</option>
        </select>
      </box-section-item>
    </section>
  </div>
</template>

<script>
import { VModel } from '~/mixins'

export default {
  mixins: [VModel],

  created() {
    $nucleus.track('PAGE_SETTINGS_NOTIFICATIONS')
  }
}
</script>

