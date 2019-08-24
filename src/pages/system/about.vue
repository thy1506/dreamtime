<template>
  <div class="about">
    <app-title>
      <h1 class="title">
        🌌 About {{ $dream.name }}
      </h1>

      <h3 class="subtitle">
        Important information, updates and community links.
      </h3>
    </app-title>

    <div class="about-body">
      <div v-if="alert" class="notification is-warning text-lg" v-html="alert" />

      <!-- Limited! -->
      <section v-if="$platform.isLimited" class="box box-section">
        <box-section-item
          label="It seems that you are offline"
          :description="`It's okay! You can use ${$dream.name} offline, but while you are disconnected we cannot offer you more information about the project, updates or report errors automatically. Connect to the Internet or if you are a developer, set the Nucleus API key and restart the program.`">
        </box-section-item>
      </section>

      <!-- Requirements -->
      <section v-if="!$platform.requirements.all" class="box box-section">
        <box-section-item icon="❗" label="Requirements" :description="`You need to meet all the requirements below to use ${$dream.name}.`" />

        <box-section-item
          label="DreamPower"
          description="DreamPower (also known as CLI) is the program necessary to transform photos. Be sure to set the correct location in the settings.">
          <template slot="icon">
            <span v-if="$platform.requirements.cli" class="item-icon">✔</span>
            <span v-else class="item-icon">❌</span>
          </template>

          <nuxt-link to="/system/settings/folders" class="button is-sm">Settings</nuxt-link>
        </box-section-item>

        <box-section-item
          label="Checkpoints"
          description="The models required by the AI. They are big but keeping them updated increases the quality of the transformation.">
          <template slot="icon">
            <span v-if="$platform.requirements.checkpoints" class="item-icon">✔</span>
            <span v-else class="item-icon">❌</span>
          </template>

          <a href="#checkpoints" class="button is-sm">Update</a>
        </box-section-item>

        <box-section-item
          label="Media Feature Pack"
          description="Multimedia package required by Windows.">
          <template slot="icon">
            <span v-if="$platform.requirements.windowsMedia" class="item-icon">✔</span>
            <span v-else class="item-icon">❌</span>
          </template>

          <app-external-link href="https://www.microsoft.com/en-us/software-download/mediafeaturepack" class="button is-sm">Download</app-external-link>
        </box-section-item>
      </section>

      <!-- DreamTime -->
      <section class="box box-section">
        <box-section-item
          :label="$dream.name"
          :description="guiDescription">
          <template slot="icon">
            <figure class="item-icon">
              <img src="~/assets/images/dreamtime.png" />
            </figure>
          </template>

          <button type="button" class="button" @click="openGUI">App Folder</button>
        </box-section-item>

        <app-update project="dreamtime" :project-title="$dream.name" />

        <box-section-item
          v-for="(item, index) in guiNavigation"
          :key="index"
          :label="item.label"
          :description="item.description"
          :icon="item.icon"
          :href="item.href"
          :version="item.version" />
      </section>

      <!-- DreamPower -->
      <section class="box box-section">
        <box-section-item
          :label="cliTitle"
          :description="cliDescription">
          <template slot="icon">
            <figure class="item-icon">
              <img src="~/assets/images/dreampower.png" />
            </figure>
          </template>

          <button type="button" class="button" @click="openCLI">App Folder</button>
        </box-section-item>

        <app-update id="checkpoints" project="checkpoints" :project-title="cliTitle" />

        <box-section-item
          v-for="(item, index) in cliNavigation"
          :key="index"
          :label="item.label"
          :description="item.description"
          :icon="item.icon"
          :href="item.href"
          :version="item.version" />
      </section>

      <!-- DreamNet -->
      <section class="box box-section">
        <box-section-item
          label="DreamNet"
          description="Community interested in developing the technology introduced by DeepNude.">
          <template slot="icon">
            <figure class="item-icon">
              <img src="~/assets/images/dreamnet.png" />
            </figure>
          </template>
        </box-section-item>

        <box-section-item
          v-for="(item, index) in dreamNetNavigation"
          :key="index"
          :label="item.label"
          :description="item.description"
          :icon="item.icon"
          :href="item.href"
          :version="item.version" />
      </section>

      <!-- Contributors -->
      <section class="box box-section is-contributors">
        <box-section-item
          icon="😍"
          label="Contributors"
          description="People who have helped us in the project either economically or solving problems in the community.">
        </box-section-item>

        <box-section-item
          v-for="(item, index) in contributors"
          :key="index"
          :label="item.name"
          :description="item.description"
          :icon="item.icon"
          :class="item.role" />
      </section>

      <!-- Developers -->
      <section class="box box-section is-contributors">
        <box-section-item
          icon="👨‍💻"
          label="Developers"
          description="Developers and important people in our projects.">
        </box-section-item>

        <box-section-item
          v-for="(item, index) in developers"
          :key="index"
          :label="item.name"
          :description="item.description"
          :icon="item.icon"
          :class="item.role">
          <div class="buttons">
            <app-external-link v-for="(link, lindex) in item.links" :key="`link-${lindex}`" :href="link.href" class="button is-sm">{{ link.name }}</app-external-link>
          </div>
        </box-section-item>
      </section>
    </div>
  </div>
</template>

<script>
import _ from 'lodash'

export default {
  computed: {
    guiNavigation() {
      return $nucleus.isEnabled ? $nucleus.about.dreamtime.navigation : []
    },

    guiDescription() {
      return $nucleus.isEnabled
        ? $nucleus.about.dreamtime.description
        : 'Friendly user interface for DreamPower.'
    },

    cliNavigation() {
      return $nucleus.isEnabled ? $nucleus.about.dreampower.navigation : []
    },

    cliTitle() {
      return $nucleus.isEnabled ? $nucleus.about.dreampower.title : 'DreamPower'
    },

    cliDescription() {
      return $nucleus.isEnabled
        ? $nucleus.about.dreampower.description
        : 'Deep learning algorithm capable of nudify people photos.'
    },

    dreamNetNavigation() {
      return $nucleus.isEnabled ? $nucleus.about.dreamnet.navigation : []
    },

    contributors() {
      return $nucleus.isEnabled ? $nucleus.about.contributors : []
    },

    developers() {
      return $nucleus.isEnabled ? $nucleus.about.developers : []
    },

    alert() {
      return $nucleus.isEnabled ? $nucleus.alerts.about : undefined
    }
  },

  created() {
    $nucleus.track('PAGE_ABOUT')
  },

  methods: {
    openGUI() {
      $tools.shell.openItem($tools.paths.getGui())
    },

    openCLI() {
      $tools.shell.openItem($tools.paths.getCli())
    },

    isURL(str) {
      if (_.isNil(str)) {
        return false
      }

      const pattern = new RegExp(
        '^(https?:\\/\\/)?' + // protocol
        '((([a-z\\d]([a-z\\d-]*[a-z\\d])*)\\.)+[a-z]{2,}|' + // domain name
        '((\\d{1,3}\\.){3}\\d{1,3}))' + // OR ip (v4) address
        '(\\:\\d+)?(\\/[-a-z\\d%_.~+]*)*' + // port and path
        '(\\?[;&a-z\\d%_.~+=-]*)?' + // query string
          '(\\#[-a-z\\d_]*)?$',
        'i'
      ) // fragment locator
      return !!pattern.test(str)
    }
  }
}
</script>

<style lang="scss">
.about {
  .about-body {
    @apply p-5;
  }

  .is-contributors {
    max-height: 400px;
    overflow-y: auto;

    .is-gold {
      .item-label {
        @apply font-bold;
        color: #d5ad6d; /*if no support for background-clip*/

        background: -webkit-linear-gradient(transparent, transparent),
          -webkit-linear-gradient(top, rgba(213, 173, 109, 1) 0%, rgba(
                  213,
                  173,
                  109,
                  1
                )
                26%, rgba(226, 186, 120, 1) 35%, rgba(163, 126, 67, 1) 45%, rgba(
                  145,
                  112,
                  59,
                  1
                )
                61%, rgba(213, 173, 109, 1) 100%);
        background-clip: text;
        -webkit-text-fill-color: transparent;
      }
    }
  }
}
</style>
