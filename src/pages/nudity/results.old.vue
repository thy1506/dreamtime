<template>
  <div class="nudity-results">
    <app-title>
      <h1 class="title">
        📷 Transformation in {{ $nudity.transformation.duration }}s
      </h1>

      <h3 class="subtitle">
        Remember to visit our server in <nuxt-link to="/about">Discord</nuxt-link> to share and discuss changes
      </h3>
    </app-title>

    <div class="content-body">
      <nudity-preview :width="412" :height="412" type="output" />

      <div class="buttons">
        <button type="button" class="button is-success" @click.prevent="save">Save</button>
        <br /><br />
        <button type="button" class="button is-primary" @click.prevent="openDirectory">Open folder</button>
        <nuxt-link to="/" class="button is-danger">Another one</nuxt-link>
      </div>
    </div>
  </div>
</template>

<script>
import _ from 'lodash'
// import download from 'downloadjs'
import moment from 'moment'

export default {
  middleware: 'nudity',

  data: () => ({}),

  methods: {
    async save() {
      const outputDataURL = await this.$nudity.modelPhoto
        .getOutputFile()
        .readAsDataURL()

      const filename = this.$nudity.modelPhoto.getOutputFileName()

      download(outputDataURL, filename, 'image/png')
    },

    openDirectory() {
      window.deepTools.shell.openItem(this.$nudity.modelPhoto.getFolderPath())
    }
  }
}
</script>

<style lang="scss">
.nudity-results {
  .buttons {
    @apply text-center;
  }
}
</style>
