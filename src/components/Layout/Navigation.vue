<template>
  <div class="layout-navbar">
    <div class="navbar" :class="{ 'is-active': isActive }">
      <!-- Welcome! -->
      <div class="navbar-header">
        <h1 class="header-title">
          {{ $dream.name }}
        </h1>

        <h3 class="header-greetings">
          {{ greetings }}
        </h3>
      </div>

      <!-- App Navigation -->
      <section class="navbar-section">
        <nav class="navbar-items">
          <nuxt-link v-if="$platform.requirements.all" to="/" class="navbar-item">
            <span class="icon">📷</span>
            <span>Nudify</span>
          </nuxt-link>

          <nuxt-link to="/system/about" class="navbar-item">
            <span class="icon">🌌</span>
            <span>About</span>
          </nuxt-link>

          <nuxt-link to="/system/settings/processing" class="navbar-item">
            <span class="icon">🔧</span>
            <span>Settings</span>
          </nuxt-link>
        </nav>
      </section>

      <!-- Developer Navigation -->
      <section v-if="isDev" class="navbar-section">
        <nav class="navbar-items">
          <a href="#" class="navbar-item" @click.prevent="testBug">
            <span class="icon">🐛</span>
            <span>I am a error!</span>
          </a>
        </nav>
      </section>
    </div>
  </div>
</template>

<script>
import moment from 'moment'

export default {
  computed: {
    greetings() {
      const hours = moment().hours()

      if (hours >= 6 && hours <= 11) {
        return 'Good morning ☕'
      }

      if (hours >= 12 && hours <= 19) {
        return 'Good afternoon 🌞'
      }

      if (hours >= 0 && hours <= 5) {
        return 'Sweet dreams 🐏'
      }

      return 'Good night 🌛'
    },

    isDev() {
      return process.env.NODE_ENV === 'development'
    },

    isActive() {
      return $settings._settings.welcome !== true
    }
  },

  methods: {
    testBug() {
      $tools.testError()
      throw new Error('wow much error')
    }
  }
}
</script>

<style lang="scss">
@keyframes bgAnim {
  0% {
    background-position: 0% 0%;
  }
  50% {
    background-position: 100% 0%;
  }
  100% {
    background-position: 0% 0%;
  }
}

@keyframes navShowAnim {
  0% {
    left: -200px;
  }

  100% {
    left: 0;
  }
}

.layout-navbar {
  @apply pb-6 shadow h-screen bg-dark relative;
  width: 200px;

  .navbar {
    @apply absolute top-0 bottom-0;
    left: -200px;
    width: inherit;

    &.is-active {
      animation-name: navShowAnim;
      animation-fill-mode: forwards;
      animation-duration: 0.5s;
      animation-timing-function: ease-in-out;
    }
  }

  .navbar-header {
    @apply mb-5 text-gray-300 flex flex-col items-center justify-center;
    animation: 20s ease-in-out infinite bgAnim;
    height: 90px;

    background: rgb(99, 66, 245);

    background: linear-gradient(
      40deg,
      rgba(99, 66, 245, 1) 0%,
      rgba(239, 125, 199, 1) 100%
    );

    background-size: 200% 100%;

    /*
    clip-path: polygon(
      50% 0%,
      100% 0,
      100% 85%,
      75% 100%,
      25% 100%,
      0 85%,
      0 0
    );
    */

    .header-title {
      @apply text-white text-xl font-bold;
    }

    .header-greetings {
      @apply text-sm;
    }
  }

  .navbar-section {
    @apply mb-5;
  }

  .section-title {
    @apply text-center font-bold;
  }

  .navbar-items {
    .navbar-item {
      @apply border-l-4 border-transparent pl-4 font-semibold flex items-center;
      height: 50px;
      transition: all 0.1s ease-in-out;

      .icon {
        @apply text-center mr-2;
        filter: grayscale(100%);
        width: 22px;
        transition: all 0.1s ease-in-out;
      }

      &:hover,
      &.nuxt-link-exact-active {
        @apply text-primary border-primary;

        .icon {
          filter: unset;
        }
      }
    }
  }
}
</style>
