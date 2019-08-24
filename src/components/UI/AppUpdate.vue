<template>
  <!-- Cannot update, only show the version... -->
  <box-section-item
    v-if="!updater.enabled"
    :label="updater.getCurrentVersion()"
    icon="🌐" />

  <!-- Updated! -->
  <box-section-item
    v-else-if="!updater.available"
    :label="`${projectTitle} it's updated!`"
    :description="`v${updater.getCurrentVersion()}`"
    icon="🌐" />

  <!-- Update available -->
  <box-section-item
    v-else-if="!updater.updating.active"
    :label="`Update available: ${updater.latest.tag_name}`"
    description="Click 'Update' to start the automatic update!"
    icon="🌐"
    class="update-item">
    <button type="button" class="button is-sm" @click.prevent="updater.download()">Update</button>
    <app-external-link v-tooltip="'Download and install the update manually.'" :href="downloadURL" class="button is-sm">Download</app-external-link>
    <button v-tooltip="'Open download folder, if you have already downloaded the update you can find it here.'" type="button" class="button is-sm" @click.prevent="openDownload">Folder</button>
  </box-section-item>

  <!-- Updating... -->
  <box-section-item
    v-else
    :label="updater.updating.text"
    icon="🌐">
    <template slot="description">
      <p class="item-description"><strong>{{ updater.updating.progress | progress }}</strong> - {{ updater.updating.mbWritten | size }}/{{ updater.updating.mbTotal | size }} MB - (Please do not close the program or leave this section)</p>
    </template>

    <button type="button" class="button is-danger is-sm" @click.prevent="updater.cancel()">Cancel</button>
  </box-section-item>
</template>

<script>
export default {
  filters: {
    progress(value) {
      value = (value * 100).toFixed(2)
      // value = Math.round(value * 100)
      return `${value}%`
    },

    size(value) {
      return value.toFixed(2)
    }
  },
  props: {
    project: {
      type: String,
      required: true
    },

    projectTitle: {
      type: String,
      default: 'Project'
    }
  },

  computed: {
    updater() {
      return this.$updater[this.project]
    },

    downloadURL() {
      return this.updater.getUpdateDownloadURLs()[0]
    }
  },

  beforeDestroy() {
    this.updater.cancel()
  },

  methods: {
    openDownload() {
      $tools.shell.openItem($tools.paths.get('downloads'))
    }
  }
}
</script>

<style lang="scss">
@keyframes updateAnim {
  0% {
    @apply bg-transparent;
  }

  50% {
    @apply bg-dark-400;
  }

  100% {
    @apply bg-transparent;
  }
}

.update-item {
  transition: all 0.2s ease-in-out;
  animation-name: updateAnim;
  animation-duration: 2s;
  animation-iteration-count: infinite;

  &:hover {
    animation-name: none;
    background: theme('colors.dark.400') !important;
  }

  .item-label {
    @apply text-white;
  }

  .item-description {
    @apply text-generic-600;
  }
}
</style>
