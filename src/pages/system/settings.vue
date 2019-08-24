<template>
  <div class="settings">
    <app-title>
      <h1 class="title">
        🔧 Settings
      </h1>

      <h3 class="subtitle">
        Customize the way {{ $dream.name }} works.
      </h3>
    </app-title>

    <div class="content-body">
      <div class="buttons is-group is-center">
        <nuxt-link to="/system/settings/processing" class="button is-outlined is-sm">Processing</nuxt-link>
        <nuxt-link to="/system/settings/preferences" class="button is-outlined is-sm">Preferences</nuxt-link>
        <nuxt-link to="/system/settings/notifications" class="button is-outlined is-sm">Notifications</nuxt-link>
        <nuxt-link to="/system/settings/folders" class="button is-outlined is-sm">Folders</nuxt-link>
        <!--<nuxt-link to="/settings/theme" class="button is-outlined is-sm">Theme</nuxt-link>-->
        <nuxt-link to="/system/settings/telemetry" class="button is-outlined is-sm">Telemetry</nuxt-link>
      </div>

      <nuxt-child v-model="settings" />
    </div>
  </div>
</template>

<script>
export default {
  data: () => ({
    settings: {}
  }),

  watch: {
    settings: {
      handler: value => {
        $settings.set(value)
      },
      deep: true
    }
  },

  created() {
    this.settings = $settings.get()
  }
}
</script>

<style lang="scss">
.settings {
  .buttons {
  }

  .settings-fields {
    @apply py-6;
  }

  .settings-fields-section {
    &:not(:last-child) {
      @apply border-b border-gray-200;
    }
  }

  .settings-field {
    @apply flex;
  }
}
</style>
