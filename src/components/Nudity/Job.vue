<template>
  <div class="c-nudity-job">
    <figure class="__preview">
      <img v-if="job.hasFinished" :src="outputDataURL" />
      <span v-else v-tooltip="'Loading...'">💭</span>
    </figure>

    <div class="__content">
      <!-- Actions -->
      <details class="__section" open>
        <summary class="__title">Actions</summary>

        <div class="__buttons">
          <button v-if="job.hasFinished" class="button is-success" @click.prevent="save">Save</button>
          <button v-if="job.hasFinished || job.hasFailed" class="button is-danger is-sm" @click.prevent="rerun">Rerun</button>
        </div>
      </details>

      <details v-if="job.hasFinished || job.isLoading" class="__section" open>
        <summary class="__title">Duration</summary>

        <div class="__status">
          <p>{{ job.timer.duration }}s</p>
        </div>
      </details>

      <details v-else-if="job.hasFailed" class="__section" open>
        <summary class="__title">Status</summary>

        <div class="__status text-danger">
          <p>Fail</p>
        </div>
      </details>

      <details v-else class="__section" open>
        <summary class="__title">Status</summary>

        <div class="__status">
          <p>Pending...</p>
        </div>
      </details>

      <!-- Preferences -->
      <details class="__section">
        <summary class="__title">Preferences</summary>

        <div class="__preferences">
          <p>
            <span class="__name">Boobs size</span>
            <span class="__value">{{ job.preferences.boobs.size | size }}</span>
          </p>

          <p>
            <span class="__name">Areola size</span>
            <span class="__value">{{ job.preferences.areola.size | size }}</span>
          </p>

          <p>
            <span class="__name">Nipple size</span>
            <span class="__value">{{ job.preferences.nipple.size | size }}</span>
          </p>

          <p>
            <span class="__name">Vagina size</span>
            <span class="__value">{{ job.preferences.vagina.size | size }}</span>
          </p>

          <p>
            <span class="__name">Pubic Hair size</span>
            <span class="__value">{{ job.preferences.pubicHair.size | size }}</span>
          </p>
        </div>
      </details>

      <!-- Console -->
      <details class="__section">
        <summary class="__title">Console</summary>

        <div class="__console">
          <li v-for="(item, index) in job.cli.lines" :key="index" :class="item.css">> {{ item.text }}</li>
        </div>
      </details>
    </div>
  </div>
</template>

<script>
import _ from 'lodash'

export default {
  filters: {
    size(value) {
      return Number.parseFloat(value).toFixed(2)
    }
  },
  props: {
    job: {
      type: Object,
      required: true
    }
  },

  data: () => ({
    outputDataURL: undefined
  }),

  watch: {
    async 'job.hasFinished'(value) {
      if (value) {
        this.outputDataURL = await this.job.getFile().readAsDataURL()
      }
    }
  },

  methods: {
    view() {},

    save() {
      const savePath = $tools.shell.showSaveDialog({
        defaultPath: this.job.getFileName(),
        filters: [
          { name: 'PNG', extensions: ['png'] }
          // { name: 'JPEG', extensions: ['jpg'] }
        ]
      })

      if (_.isNil(savePath)) {
        return
      }

      this.job.getFile().copy(savePath)
    },

    rerun() {
      this.job.photo.rerunJob(this.job.id)
    }
  }
}
</script>

<style lang="scss">
.c-nudity-job {
  @apply flex;

  .__preview {
    @apply flex justify-center items-center
        rounded rounded-tr-none rounded-br-none
        border-2 border-dark border-r-0
        text-3xl;
    width: 125px;
    height: 125px;

    img {
      @apply w-full h-full rounded rounded-tr-none rounded-br-none;
      transition: all 0.15s ease-in-out;

      &:hover {
        @apply rounded z-50;
        transform: scale(3);
      }
    }
  }

  .__content {
    @apply flex-1 flex flex-col
        bg-dark
        rounded
        rounded-tl-none
        rounded-bl-none
        px-4
        shadow;

    width: 200px;

    .__section {
      @apply py-3;

      &:not(:last-child) {
        @apply border-b border-dark-400;
      }

      .__title {
        @apply text-xs uppercase text-generic-300 mb-2 cursor-pointer;
      }

      .__status {
        @apply flex justify-center
          text-xl font-bold;
      }

      .__buttons {
        @apply flex justify-center items-center;
      }

      .__console {
        @apply p-3 bg-black overflow-auto rounded;
        height: 150px;

        li {
          @apply font-mono text-xs text-generic-100 mb-3 block;

          &.text-danger {
            @apply text-danger;
          }
        }
      }

      .__preferences {
        p {
          @apply text-sm;

          .__name {
            @apply inline-block text-generic-300;
            width: 150px;
          }

          .__value {
            @apply inline-block font-bold text-generic-100;
          }
        }
      }
    }
  }
}

/*
.c-nudity-job {
  @apply flex pb-4;
  height: 170px;

  &:not(:first-child) {
    @apply pt-4;
  }

  &:not(:last-child) {
    @apply border-b border-dark-400;
  }

  .job-section {
    @apply p-2 mb-0;
    width: 200px;

    &:not(:last-child) {
      @apply mr-5;
    }
  }

  .job-photos {
    @apply flex-1 inline-flex;
    width: auto;
  }

  .job-status {
    @apply flex flex-col justify-center items-center;
    width: 150px;

    & > div {
      @apply text-center;
    }

    .buttons {
      @apply flex flex-col justify-center items-center mt-3;
    }

    .status-title {
      @apply font-semibold text-xl text-generic-300;

      &.text-danger {
        @apply text-danger;
      }
    }

    .status-text {
      @apply text-sm;
    }
  }

  .job-console {
    @apply bg-black overflow-auto rounded;

    li {
      @apply font-mono text-xs text-generic-100 mb-3 block;

      &.text-danger {
        @apply text-danger;
      }
    }
  }

  .job-preferences {
    @apply flex flex-col justify-center items-center;

    p {
      @apply text-sm;

      .preference-name {
        @apply inline-block;
        width: 150px;
      }

      .preference-value {
        @apply inline-block;
        @apply font-bold;
      }
    }
  }
}
*/
</style>
