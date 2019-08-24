// See default config https://github.com/tailwindcss/tailwindcss/blob/master/stubs/defaultConfig.stub.js
module.exports = {
  theme: {
    extend: {
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          'Nunito',
          'Nunito Sans',
          'system-ui',
          'Helvetica Neue',
          'Arial',
          'sans-serif'
        ]
      },

      colors: {
        generic: {
          100: '#DAD9DE',
          200: '#C2C1C8',
          300: '#A9A8B2',
          400: '#918F9C',
          default: '#797786',
          600: '#6E6D7A',
          700: '#64626E',
          800: '#595762',
          900: '#4E4C56'
        },

        dark: {
          100: '#7A7884',
          200: '#676473',
          300: '#545161',
          400: '#413E50',
          default: '#2f2b3f',
          600: '#2B283A',
          700: '#272434',
          800: '#231f2f',
          900: '#1E1C29'
        },

        blue: {
          default: '#2eb9e7'
        },

        violet: {
          default: '#7020ea'
        },

        pink: {
          default: '#ef98d0'
        },

        danger: {
          10: 'rgba(244, 67, 54, 0.10)',
          15: 'rgba(244, 67, 54, 0.15)',
          20: 'rgba(244, 67, 54, 0.20)',
          25: 'rgba(244, 67, 54, 0.25)',
          default: '#f44336'
        },

        success: {
          10: 'rgba(87, 184, 70, 0.10)',
          20: 'rgba(87, 184, 70, 0.20)',
          default: '#57b846'
        },

        warning: '#c17c16',

        primary: {
          10: 'rgba(236, 110, 129, 0.10)',
          15: 'rgba(236, 110, 129, 0.15)',
          20: 'rgba(236, 110, 129, 0.20)',
          25: 'rgba(236, 110, 129, 0.25)',

          100: '#F195A3',
          200: '#EF8897',
          300: '#ED7B8C',
          default: '#ec6e81',
          500: '#D76476',
          600: '#C25B6A',
          700: '#AC515E',
          800: '#974653',
          900: '#813C47'
        }
      }
    }
  },
  variants: {}
}
