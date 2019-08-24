<template>
  <div class="c-settings-preferences">
    <section class="box box-section">
      <box-section-item
        label="Number of runs"
        description="The transformation can be different each time, here you can set how many times your photo will be processed. In the end you can save the one you like best.">
        <input v-model="currentValue.executions" type="number" min="1" class="input" />
      </box-section-item>

      <box-section-item
        label="Randomize every run"
        description="Random preferences will be set at each run">
        <select v-model="currentValue.randomizePreferences" class="input">
          <option :value="true">Enabled</option>
          <option :value="false">Disabled</option>
        </select>
      </box-section-item>

      <box-section-item
        v-show="!currentValue.randomizePreferences"
        label="Progressive every run"
        description="Preferences will increase their value +0.2 in each run">
        <select v-model="currentValue.progressivePreferences" class="input">
          <option :value="true">Enabled</option>
          <option :value="false">Disabled</option>
        </select>
      </box-section-item>
    </section>

    <section class="box box-section">
      <box-section-item label="Boob Size" :description="`Current value: ${currentValue.boobs.size}`">
        <div class="slider-container">
          <input
            v-model="currentValue.boobs.size"
            type="range"
            class="slider"
            in="0.3"
            max="2"
            step="0.1" />
          <span class="min">0.3</span>
          <span class="max">2.0</span>
        </div>
      </box-section-item>

      <box-section-item
        v-show="currentValue.randomizePreferences"
        label="Randomize"
        description="Randomize the value in each run.">
        <select v-model="currentValue.boobs.randomize" class="input">
          <option :value="true">Enabled</option>
          <option :value="false">Disabled</option>
        </select>
      </box-section-item>

      <box-section-item
        v-show="!currentValue.randomizePreferences && currentValue.progressivePreferences"
        label="Progressive"
        description="Increase the value progressively in each run">
        <select v-model="currentValue.boobs.progressive" class="input">
          <option :value="true">Enabled</option>
          <option :value="false">Disabled</option>
        </select>
      </box-section-item>
    </section>

    <section class="box box-section">
      <box-section-item label="Areola Size" :description="`Current value: ${currentValue.areola.size}`">
        <div class="slider-container">
          <input
            v-model="currentValue.areola.size"
            type="range"
            class="slider"
            in="0.3"
            max="2"
            step="0.1" />
          <span class="min">0.3</span>
          <span class="max">2.0</span>
        </div>
      </box-section-item>

      <box-section-item
        v-show="currentValue.randomizePreferences"
        label="Randomize"
        description="Randomize the value in each run.">
        <select v-model="currentValue.areola.randomize" class="input">
          <option :value="true">Enabled</option>
          <option :value="false">Disabled</option>
        </select>
      </box-section-item>

      <box-section-item
        v-show="!currentValue.randomizePreferences && currentValue.progressivePreferences"
        label="Progressive"
        description="Increase the value progressively in each run">
        <select v-model="currentValue.areola.progressive" class="input">
          <option :value="true">Enabled</option>
          <option :value="false">Disabled</option>
        </select>
      </box-section-item>
    </section>

    <section class="box box-section">
      <box-section-item label="Nipple Size" :description="`Current value: ${currentValue.nipple.size}`">
        <div class="slider-container">
          <input
            v-model="currentValue.nipple.size"
            type="range"
            class="slider"
            in="0.3"
            max="2"
            step="0.1" />
          <span class="min">0.3</span>
          <span class="max">2.0</span>
        </div>
      </box-section-item>

      <box-section-item
        v-show="currentValue.randomizePreferences"
        label="Randomize"
        description="Randomize the value in each run.">
        <select v-model="currentValue.nipple.randomize" class="input">
          <option :value="true">Enabled</option>
          <option :value="false">Disabled</option>
        </select>
      </box-section-item>

      <box-section-item
        v-show="!currentValue.randomizePreferences && currentValue.progressivePreferences"
        label="Progressive"
        description="Increase the value progressively in each run">
        <select v-model="currentValue.nipple.progressive" class="input">
          <option :value="true">Enabled</option>
          <option :value="false">Disabled</option>
        </select>
      </box-section-item>
    </section>

    <section class="box box-section">
      <box-section-item label="Vagina Size" :description="`Current value: ${currentValue.vagina.size}`">
        <div class="slider-container">
          <input
            v-model="currentValue.vagina.size"
            type="range"
            class="slider"
            in="0.3"
            max="1.5"
            step="0.1"></input>
          <span class="min">0.3</span>
          <span class="max">1.5</span>
        </div>
      </box-section-item>

      <box-section-item
        v-show="currentValue.randomizePreferences"
        label="Randomize"
        description="Randomize the value in each run.">
        <select v-model="currentValue.vagina.randomize" class="input">
          <option :value="true">Enabled</option>
          <option :value="false">Disabled</option>
        </select>
      </box-section-item>

      <box-section-item
        v-show="!currentValue.randomizePreferences && currentValue.progressivePreferences"
        label="Progressive"
        description="Increase the value progressively in each run">
        <select v-model="currentValue.vagina.progressive" class="input">
          <option :value="true">Enabled</option>
          <option :value="false">Disabled</option>
        </select>
      </box-section-item>
    </section>

    <section class="box box-section">
      <box-section-item label="Pubic Hair" :description="`Current value: ${currentValue.pubicHair.size}`">
        <div class="slider-container">
          <input
            v-model="currentValue.pubicHair.size"
            type="range"
            class="slider"
            in="0"
            max="2"
            step="0.1"></input>
          <span class="min">Disabled</span>
          <span class="max">2.0</span>
        </div>
      </box-section-item>

      <box-section-item
        v-show="currentValue.randomizePreferences"
        label="Randomize"
        description="Randomize the value in each run.">
        <select v-model="currentValue.pubicHair.randomize" class="input">
          <option :value="true">Enabled</option>
          <option :value="false">Disabled</option>
        </select>
      </box-section-item>

      <box-section-item
        v-show="!currentValue.randomizePreferences && currentValue.progressivePreferences"
        label="Progressive"
        description="Increase the value progressively in each run">
        <select v-model="currentValue.pubicHair.progressive" class="input">
          <option :value="true">Enabled</option>
          <option :value="false">Disabled</option>
        </select>
      </box-section-item>
    </section>

    <section class="box box-section">
      <box-section-item v-if="false" label="Photo restoration" description="Restore the cropped photo to the original photo.">
        <select v-model="currentValue.useRestoration" class="input">
          <option :value="true">Enabled</option>
          <option :value="false">Disabled</option>
        </select>
      </box-section-item>

      <box-section-item v-if="false" label="waifu2x" description="waifu2x will try to resize your transformed photo to 1024x1024 with the least possible quality loss.">
        <select v-model="currentValue.useWaifu" class="input">
          <option :value="true">Enabled</option>
          <option :value="false">Disabled</option>
        </select>
      </box-section-item>

      <box-section-item
        v-if="false"
        label="Use Custom Mask"
        hint="(Advanced) You can edit the masks of the photo before processing.">
        <select v-model="currentValue.useCustomMask" class="input">
          <option :value="false">Disabled</option>
          <option :value="true">Enabled</option>
        </select>
      </box-section-item>
    </section>
  </div>
</template>

<script>
import { VModel } from '~/mixins'

export default {
  mixins: [VModel]
}
</script>

