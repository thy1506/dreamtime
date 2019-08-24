<template>
  <div class="app-title">
    <slot />
  </div>
</template>

<style lang="scss">
.app-title {
  @apply px-4 py-2 bg-dark shadow flex flex-col justify-center;
  height: 90px;

  .title {
    @apply font-bold text-white text-xl;

    sup {
      @apply text-white;
    }
  }

  .subtitle {
    @apply text-lg text-generic-300;
  }

  a {
    @apply underline;
  }
}
</style>
