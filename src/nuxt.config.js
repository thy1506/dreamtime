require('dotenv').config()

module.exports = {
  mode: 'spa',

  /**
   *
   */
  router: {
    mode: 'hash'
  },

  /**
   * Dev-Server settings
   */
  server: {
    port: process.env.SERVER_PORT,
    host: process.env.SERVER_HOST
  },

  /*
   ** Headers of the page
   */
  head: {
    title: `${process.env.APP_NAME} v${process.env.APP_VERSION}`,

    meta: [
      { charset: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' }
    ],

    link: [
      /* { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' } */
    ],

    scripts: [
      /*
      {
        src: 'https://www.gstatic.com/firebasejs/6.3.1/firebase-app.js'
      }
      */
    ]
  },

  /*
   ** Customize the progress-bar color
   */
  loading: { color: '#fff' },

  /*
   ** Global CSS
   */
  css: [
    '~/assets/css/tailwind.scss',
    '~/assets/css/fonts.scss',
    '~/assets/css/vendor.scss'
  ],

  /*
   ** Plugins to load before mounting the App
   */
  plugins: ['~/plugins/boot.client.js', '~/components'],

  /*
   ** Nuxt.js modules
   */
  modules: [
    '@nuxtjs/pwa',
    // '@nuxtjs/eslint-module',
    '@nuxtjs/dotenv'
  ],

  /*
   ** Axios module configuration
   ** See https://axios.nuxtjs.org/options
   */
  axios: {},

  /**
   *
   */
  dev: process.env.NODE_ENV === 'development',

  /**
   *
   */
  env: {
    APP_NAME: process.env.APP_NAME,
    APP_VERSION: process.env.APP_VERSION,
    NUCLEUS_APPID: process.env.NUCLEUS_APPID
  },

  /*
   ** Build configuration
   */
  build: {
    /**
     *
     */
    postcss: {
      plugins: {
        tailwindcss: './tailwind.config.js'
      }
    },

    /**
     *
     */
    terser: {
      parallel: true,
      cache: false,
      sourceMap: true,
      extractComments: {
        filename: 'LICENSES'
      },
      terserOptions: {
        output: {
          comments: /^\**!|@preserve|@license|@cc_on/
        }
      }
    },

    /*
     ** You can extend webpack config here
     */
    extend(config, { isClient, isDev }) {
      console.log({
        isClient,
        isDev
      })

      config.output.futureEmitAssets = false
      config.devtool = isClient ? 'source-map' : 'inline-source-map'

      if (!isDev) {
        // const RollbarSourceMapPlugin = require('rollbar-sourcemap-webpack-plugin')
        /*
        config.plugins.push(
          new RollbarSourceMapPlugin({
            accessToken: process.env.ROLLBAR_ACCESS_TOKEN,
            version: process.env.npm_package_version,
            publicPath: source => {
              console.log(source)
              return `@/${source}`
            }
          })
        )
        */
        /*
        config.plugins.push(
          new SentryWebpackPlugin({
            include: '.',
            ignoreFile: '.sentrycliignore',
            ignore: ['node_modules', 'nuxt.config.js'],
            configFile: 'sentry.properties'
          })
        )
        */
      }

      if (!isDev) {
        // relative links, please.
        config.output.publicPath = './_nuxt/'
      }
    }
  }
}
