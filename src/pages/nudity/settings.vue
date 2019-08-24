<template>
  <div class="nudity-settings">
    <div class="content-body">
      <nudity-preview class="mb-5" type="cropped" />

      <div class="settings-fields">
        <section class="settings-fields-section">
          <form-inline-field
            label="Number of executions"
            hint="The transformation process will be repeated this number of times and you can select the one that has the best result.">
            <input v-model="preferences.executions" type="number" min="1" class="input" />
          </form-inline-field>

          <form-inline-field
            label="Boobs Size"
            hint="The algorithm will try to create the indicated size">
            <select v-model="preferences.boobsSize" class="input">
              <option value="small">Small</option>
              <option value="medium">Medium</option>
              <option value="big">Big</option>
            </select>
          </form-inline-field>

          <form-inline-field
            label="Pubic Hair Size"
            hint="The algorithm will try to create the indicated size">
            <select v-model="preferences.pubicHairSize" class="input">
              <option value="none">None</option>
              <option value="small">Small</option>
              <option value="medium">Medium</option>
              <option value="big">Big</option>
            </select>
          </form-inline-field>

          <form-inline-field
            label="Use Custom Mask"
            hint="(Advanced) You can edit the masks of the photo before processing.">
            <select v-model="preferences.useCustomMask" class="input">
              <option :value="false">Disabled</option>
              <option :value="true">Enabled</option>
            </select>
          </form-inline-field>
        </section>

        <div class="buttons">
          <nuxt-link to="/nudity/crop" class="button is-danger">Back</nuxt-link>
          <nuxt-link to="/nudity/results" class="button is-success">Nudify!</nuxt-link>
        </div>
      </div>

      <!-- Loading
      <div v-else class="settings-loading">
        <h1 class="title">{{ $nudity.transformation.duration }}s</h1>

        <h1 class="title">🧜‍ Loading...</h1>

        <h3 class="subtitle">Transforming your photo with the power of your {{ deviceName }}!</h3>

        <div ref="cli" class="settings-cli">
          <p v-for="(line, index) in $nudity.modelPhoto.cliLines" :key="index" class="cli-line">{{ line }}</p>
        </div>
      </div>
      -->
    </div>
  </div>
</template>

<script>
export default {
  middleware: 'nudity',

  data: () => ({
    preferences: {}
  }),

  created() {
    this.preferences = this.$nudity.modelPhoto.transformation.preferences
  }
}
</script>

<style lang="scss">
.nudity-settings {
  .settings-fields {
  }

  .buttons {
    @apply mt-5;
  }

  .settings-loading {
    @apply text-center;

    .title {
      @apply font-bold text-3xl;
    }

    .subtitle {
      @apply text-gray-600 text-xl;
    }
  }

  .settings-cli {
    @apply mt-6 bg-black p-4 w-full font-mono overflow-y-auto;
    height: 150px;

    .cli-line {
      @apply block text-white;
    }
  }
}
</style>
