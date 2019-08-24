/*
 * DreamTime | (C) 2019 by Ivan Bravo Bravo <ivan@dreamnet.tech>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License 3.0 as published by
 * the Free Software Foundation.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

import Vue from 'vue'
import Title from './AppTitle'
import ExternalLink from './AppExternalLink'
import Update from './AppUpdate'
import SectionItem from './BoxSectionItem'
import AppPhoto from './AppPhoto'

Vue.component('app-title', Title)
Vue.component('app-external-link', ExternalLink)
Vue.component('app-update', Update)
Vue.component('app-photo', AppPhoto)
Vue.component('box-section-item', SectionItem)
