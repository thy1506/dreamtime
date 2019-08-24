<template>
  <div class="settings-fields">
    <p class="notification">If you change any of these options please restart the program to avoid problems.</p>

    <section class="box box-section">
      <box-section-item
        label="DreamPower"
        description="Location of DreamPower (also known as CLI)">
        <input v-model="currentValue.folders.cli" class="input" />
        <button type="button" class="button" @click.prevent="changeCLI">📂</button>
      </box-section-item>

      <box-section-item
        label="Models"
        description="Location where the transformed photos will be saved.">
        <input v-model="currentValue.folders.models" class="input" />
        <button type="button" class="button" @click.prevent="changeModels">📂</button>
      </box-section-item>

      <box-section-item
        label="Cropped"
        description="Location where the cropped photos will be saved. We recommend selecting a temporary folder.">
        <input v-model="currentValue.folders.cropped" class="input" />
        <button type="button" class="button" @click.prevent="changeCropped">📂</button>
      </box-section-item>

      <box-section-item
        v-if="false"
        label="Masks"
        description="Location where the algorithm masks photos will be saved.">
        <input v-model="currentValue.folders.masks" class="input" />
        <button type="button" class="button" @click.prevent="changeMasks">📂</button>
      </box-section-item>
    </section>
  </div>
</template>

<script>
import _ from 'lodash'
import { VModel } from '~/mixins'

export default {
  mixins: [VModel],

  data: () => ({}),

  created() {
    $nucleus.track('PAGE_SETTINGS_FOLDERS')
  },

  methods: {
    showOpenDialog(path) {
      const dir = $tools.shell.showOpenDialog(null, {
        defaultPath: path,
        properties: ['openDirectory']
      })

      if (_.isNil(dir)) {
        return path
      }

      if (!$tools.fs.exists(dir[0])) {
        // ???
        return path
      }

      return dir[0]
    },

    changeModels() {
      const dir = this.showOpenDialog(this.currentValue.folders.models)
      this.currentValue.folders.models = dir
    },

    changeCropped() {
      const dir = this.showOpenDialog(this.currentValue.folders.cropped)
      this.currentValue.folders.cropped = dir
    },

    changeMasks() {
      const dir = this.showOpenDialog(this.currentValue.folders.masks)
      this.currentValue.folders.masks = dir
    },

    changeCLI() {
      const dir = this.showOpenDialog(this.currentValue.folders.cli)
      this.currentValue.folders.cli = dir
    }
  }
}
</script>

<style lang="scss" scoped>
p {
  @apply mb-5;
}
</style>


