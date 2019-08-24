<template>
  <figure class="app-photo">
    <img :src="src" />

    <p class="app-photo-label">
      <slot />
    </p>
  </figure>
</template>

<script>
export default {
  props: {
    src: {
      type: undefined,
      required: true
    }
  }
}
</script>

<style lang="scss">
.app-photo {
  @apply relative flex flex-col;
  box-sizing: content-box;
  width: 125px;

  img {
    @apply w-full rounded-t shadow-md;
    height: 125px;
    transition: all 0.15s ease-in-out;

    &:hover {
      @apply rounded z-50;
      transform: scale(2.5);
    }
  }

  .app-photo-label {
    @apply p-1 bg-dark-600 text-generic-300 text-sm rounded-b font-semibold text-center;
  }
}
</style>
