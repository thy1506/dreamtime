<template>
  <div class="c-nudity-upload">
    <!-- Dropzone -->
    <div
      class="upload-dropzone"
      :class="{'is-dragging': isDraggingFile}"
      @dragenter="onDragEnter"
      @dragover="onDragOver"
      @dragleave="onDragLeave"
      @drop="onDrop">
      <p class="dropzone-hint">👇 Drop the photo here!</p>
    </div>

    <!-- Hidden input -->
    <input
      v-show="false"
      ref="photo"
      type="file"
      accept="image/jpeg, image/png"
      @change="onPhotoSelected" />

    <div class="box py-5">
      <div class="upload-url">
        <input v-model="webAddress" type="url" class="input" placeholder="🌍 or enter a web address..." />

        <button class="button" @click="onURL">
          Go!
        </button>
      </div>

      <!-- Action button -->
      <button class="button" @click.prevent="$refs.photo.click()">
        📂 or open a photo...
      </button>
    </div>
  </div>
</template>

<script>
import _ from 'lodash'
import swal from 'sweetalert'
import { Photo } from '~/modules/models'
import { File } from '~/modules'

export default {
  props: {
    model: {
      type: String,
      default: undefined
    }
  },

  data: () => ({
    webAddress: '',

    // Indicates if the user is dragging a file in the window (we apply the drag style)
    isDraggingFile: false
  }),

  created() {
    // Restarts the information of a previous process
    this.$nudify.reset()
  },

  methods: {
    /**
     * File selected, start a new transformation process
     */
    startFromFile(inputFile) {
      if (_.isNil(inputFile)) {
        swal(
          'Upload failed',
          'It seems that you have not selected a photo!',
          'info'
        )
        return
      }

      // New File instance
      const file = File.fromPath(inputFile.path)

      this.start(file)
    },

    /**
     *
     */
    async startFromURL(url) {
      if (_.isNil(url)) {
        swal('Upload failed', 'This does not seem like a valid URL', 'info')
        return
      }

      swal({
        title: 'Loading...',
        text: 'We are downloading the photo and preparing it!',
        button: false,
        closeOnClickOutside: false,
        closeOnEsc: false
      })

      try {
        // New File instance
        const file = await File.fromURL(url)

        swal.close()

        this.start(file)
      } catch (err) {
        swal({
          icon: 'error',
          title: 'Upload failed',
          text: `An error has occurred downloading the photo or saving it in the temporary folder, please make sure you are connected to the Internet and that ${
            $dream.name
          } has permissions to save files.`
        })

        $rollbar.warn(err)
      }
    },

    /**
     *
     */
    start(file) {
      // Create a photo for the model ("null" model for now)
      const photo = new Photo(null, file)

      // Get any error message from the file
      const validationErrorMessage = photo.getValidationErrorMessage()

      if (!_.isNil(validationErrorMessage)) {
        swal('Upload failed', validationErrorMessage, 'error')
        return
      }

      // Start the transformation process!
      this.$nudify.start(photo)

      // It's time to crop the photo
      this.$router.push('/nudity/crop')
    },

    /**
     *
     */
    onPhotoSelected(event) {
      const { files } = event.target

      if (files.length === 0) {
        return
      }

      $nucleus.track('UPLOAD_SELECTED')

      this.startFromFile(files[0])
      event.target.value = ''
    },

    /**
     *
     */
    onURL() {
      if (_.isNil(this.webAddress) || this.webAddress.length === 0) {
        swal('Upload failed', 'Please enter a valid web address', 'error')
        return
      }

      $nucleus.track('UPLOAD_URL')

      this.startFromURL(this.webAddress)
    },

    /**
     *
     */
    onDragEnter(event) {
      event.dataTransfer.dropEffect = 'copy'
      this.isDraggingFile = true
    },

    /**
     *
     */
    onDragLeave() {
      this.isDraggingFile = false
    },

    /**
     *
     */
    onDragOver(event) {
      event.preventDefault()
      event.stopPropagation()
      event.dataTransfer.dropEffect = 'copy'
      this.isDraggingFile = true
    },

    /**
     *
     */
    onDrop(event) {
      event.preventDefault()
      event.stopPropagation()
      this.isDraggingFile = false

      const { files } = event.dataTransfer
      const externalURL = event.dataTransfer.getData('url')

      if (files.length > 0) {
        $nucleus.track('UPLOAD_DROP')
        this.startFromFile(files[0])
      } else if (externalURL.length > 0) {
        $nucleus.track('UPLOAD_DROP_URL')
        this.startFromURL(externalURL)
      }
    }
  }
}
</script>

<style lang="scss">
.c-nudity-upload {
  @apply w-full
    relative
    text-center
    mb-5;

  .upload-dropzone {
    @apply flex
      items-center
      justify-center
      bg-dark-400
      rounded
      border-transparent
      border-2
      border-dashed
      mb-5;

    height: 150px;
    transition: all 0.1s linear;

    &.is-dragging {
      @apply bg-dark-700 border-white;

      .dropzone-hint {
        @apply text-white;
      }
    }

    .dropzone-hint {
      @apply text-generic-300 uppercase;
      transition: all 0.1s linear;
    }
  }

  .upload-url {
    @apply mb-5 flex;

    .input {
      @apply flex-1 mr-3;
    }
  }

  &.is-dragging {
    @apply border-white border-dotted;

    .dragging-overlay {
      //display: block;
    }
  }

  .dragging-overlay {
    @apply bg-white absolute top-0 left-0 right-0 bottom-0 hidden;
    opacity: 0.3;
  }

  .fu-hint {
    @apply text-sm text-gray-600;
  }
}
</style>


