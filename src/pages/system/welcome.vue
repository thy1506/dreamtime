<template>
  <div class="welcome">
    <h1 class="welcome-title">Welcome to {{ $dream.name }}!</h1>
    <h2 class="welcome-subtitle">Before starting we would like to remind you:</h2>

    <ul class="welcome-list">
      <li>💘 We (DreamNet) are very grateful that you use our application, we will continue working to improve it!</li>
      <li>🥇 {{ $dream.name }} is a free and open-source project. <b>Do not pay for it!</b></li>
      <li>📷 The intention of this program is to allow its users to generate photos for private entertainment as well as to demonstrate and educate people about this technology.</li>
      <li>🛑 Do not use our project to hurt people or generate problems. We totally reject this behavior, don't be a jerk and give us the opportunity to improve this technology.</li>
      <li>📃 {{ $dream.name }} and DreamPower are licensed under the GNU General Public License v3.0. TL;DR: You can share and modify the program but always as open-source.</li>
    </ul>

    <div class="welcome-buttons">
      <button type="button" class="button" @click="next">Understood!</button>
    </div>
  </div>
</template>

<script>
export default {
  methods: {
    next() {
      $settings.welcome = false
      this.$router.push('/')
    }
  }
}
</script>

<style lang="scss">
.welcome {
  @apply p-8;
  height: 100%;

  .welcome-title {
    @apply mb-1 text-white text-3xl font-bold text-center;
  }

  .welcome-subtitle {
    @apply mb-10 text-generic-600 text-2xl font-bold text-center;
  }

  .welcome-list {
    @apply mx-5 text-xl font-semibold;

    li {
      @apply mb-6 list-inside;
    }
  }

  .welcome-buttons {
    @apply text-center;
  }
}
</style>
