<template>
  <div class="nudify-results">
    <app-title>
      <h1 class="title">
        🧜‍♀️ Transforming... {{ photo.timer.duration }}s
      </h1>

      <h3 class="subtitle">
        Your photo is being transformed with the power of your {{ deviceName }}.
      </h3>
    </app-title>

    <div class="content-body">
      <!-- Stats -->
      <div class="__summary">
        <div class="__content">
          <div class="__status box">
            <p v-if="!photo.isLoading">📸 Dream completed<br>Has it been a good dream?</p>
            <p v-else>💭 Dreaming... this will take a moment</p>
          </div>

          <div class="__photos">
            <app-photo :src="sourceDataURL">Original</app-photo>
            <app-photo :src="croppedDataURL">Cropped</app-photo>
          </div>

          <div v-show="!photo.isLoading" class="__actions">
            <div class="buttons">
              <button type="button" class="button is-success" @click.prevent="togglePreferences">
                <span v-if="!isPreferencesVisible">Change Preferences</span>
                <span v-else>Results</span>
              </button>

              <button type="button" class="button is-danger" @click.prevent="rerun">Rerun all</button>
            </div>

            <div class="buttons">
              <button type="button" class="button" @click.prevent="openFolder">Open Folder</button>
              <nuxt-link to="/" class="button">Another photo</nuxt-link>
            </div>
          </div>

          <div v-show="photo.isLoading" class="__actions justify-center items-center">
            <button type="button" class="button is-danger" @click.prevent="cancel">Cancel</button>
          </div>
        </div>
      </div>

      <!-- Jobs -->
      <div v-show="!isPreferencesVisible" class="__jobs">
        <nudity-job v-for="(job, index) in photo.jobs" :key="index" :job="job" />
      </div>

      <!-- Preferences -->
      <settings-preferences v-show="isPreferencesVisible" v-model="photo.preferences" />
    </div>
  </div>
  </div>
</template>

<script>
export default {
  middleware: 'nudity',

  data: () => ({
    isPreferencesVisible: false,

    sourceDataURL: undefined,
    croppedDataURL: undefined
  }),

  computed: {
    photo() {
      return this.$nudify.photo
    },

    /**
     *
     */
    preferences() {
      return this.$nudify.photo.preferences
    },

    /**
     *
     */
    deviceName() {
      return $settings.processing.device
    }
  },

  async created() {
    this.sourceDataURL = await this.photo.getSourceFile().readAsDataURL()
    this.croppedDataURL = await this.photo.getCroppedFile().readAsDataURL()
  },

  mounted() {
    if (process.env.NODE_ENV === 'production') {
      this.photo.start()
    }
  },

  beforeDestroy() {
    this.photo.cancel()
  },

  methods: {
    /**
     *
     */
    rerun() {
      this.isPreferencesVisible = false
      this.photo.rerun()
    },

    /**
     *
     */
    cancel() {
      this.photo.cancel()
    },

    /**
     *
     */
    openFolder() {
      $tools.shell.openItem(this.photo.getFolderPath())
    },

    /**
     *
     */
    togglePreferences() {
      this.isPreferencesVisible = !this.isPreferencesVisible
    }
  }
}
</script>

<style lang="scss">
.nudify-results {
  .content-body {
    @apply flex;
  }

  .__summary {
    @apply flex-1 mr-5;

    .__content {
      @apply sticky top-0;
    }

    .__status {
      @apply flex justify-center items-center
        text-xl;
    }

    .__photos {
      @apply flex justify-center
        mb-5;

      .app-photo {
        &:not(:last-child) {
          @apply mr-5;
        }
      }
    }

    .__actions {
      @apply flex;

      .buttons {
        @apply flex-1 flex-col justify-center;
      }
    }
  }

  .__jobs {
    @apply flex-1 flex flex-col;

    .c-nudity-job {
      @apply mr-5 mb-5;
    }
  }

  .c-settings-preferences {
    @apply flex-1;
  }
}
</style>
