<template>
  <div class="c-inline-field">
    <div class="inline-field-left">
      <p class="inline-field-label">{{ label }}</p>
      <p v-if="hint" class="inline-field-hint">{{ hint }}</p>
    </div>

    <div class="inline-field-right">
      <slot />
    </div>
  </div>
</template>

<script>
export default {
  props: {
    label: {
      type: String,
      required: true
    },

    hint: {
      type: String,
      default: undefined
    }
  }
}
</script>

<style lang="scss">
.c-inline-field {
  @apply flex;

  &:not(:last-child) {
    @apply mb-5;
  }

  .inline-field-left {
    @apply flex-1 pr-6;
  }

  .inline-field-right {
    @apply flex items-center justify-end;
    width: 250px;
  }

  .inline-field-label {
    @apply font-semibold;
  }

  .inline-field-hint {
    @apply text-gray-600 text-sm;
  }
}
</style>

