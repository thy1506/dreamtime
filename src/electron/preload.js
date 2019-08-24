const { AppError, settings, nucleus, rollbar } = require('./modules')
const tools = require('./tools')

// Custom Error
window.AppError = AppError

// User settings
window.$settings = settings

// Analytics & App settings
// https://nucleus.sh/docs/gettingstarted
window.$nucleus = nucleus

// Error reporting
// https://docs.rollbar.com/docs/nodejs
window.$rollbar = rollbar

// Tools
window.$tools = tools

// Tools (Legacy)
window.deepTools = window.$tools
